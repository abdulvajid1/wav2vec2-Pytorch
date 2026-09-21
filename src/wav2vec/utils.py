from dataclasses import dataclass, asdict
from typing import Literal
from typing import Optional
import torch 
import torch.nn as nn
import soundfile as sf

import logging
import sys

GREEN = "\033[92m"
RESET = "\033[0m"

class SimpleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{GREEN}[{record.levelname}]{RESET}: {record.getMessage()}\n"

def get_logger(name: str = "wav2vec2", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SimpleFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

import warnings
warnings.filterwarnings("ignore")

@dataclass
class Wav2Vec2ForPreTrainingOutput:

    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None

@dataclass
class Wav2Vec2Config:

    ### FEATURE ENCODER CONVOLUTION CONFIG ###
    conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_strides: tuple = (5, 2, 2, 2, 2, 2, 2)
    conv_kernels: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = True
    feature_projection_dropout_p: float = 0.0

    ### POSITIONAL CONVOLUTIONAL EMBEDDING ###
    conv_positional_emb_drop_p: float = 0.0
    conv_positional_emb_groups: int = 16
    conv_positional_emb_kernel_size: int = 128

    ### TRANSFORMER CONFIG ###
    num_transformer_layers: int = 12
    num_attention_heads: int = 12
    embedding_dimension: int = 768
    mlp_ratio: int = 4
    mlp_dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    transformer_encoder_dropout: float = 0.0
    layer_dropout: float = 0.0
    initializer_range: float = 0.02

    ### GUMBEL SOFTMAX CONFIG ###
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256
    pre_quantizer_dropout: float = 0.0

    ### MASKING CONFIG ###
    masking_probability: float = 0.065
    masking_span_length: int = 10 
    minimum_spans: int = 2

    ### LOSS CONFIG ###
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    ### TRAINING CONFIG ###
    num_negatives: int = 100

    ### LayerNorm Config ###
    layer_norm_eps: float = 1e-5
    
    ### CTC Config ###
    asr_head_dropout_p: float = 0.1
    blank_token_idx: int = 0
    vocab_size: int = 32

    ### Huggingface Interface Config ###
    hf_model_name: str = "facebook/wav2vec2-base"

    ### Pretrain Backbone Config ###
    path_to_pretrained_weights: str | None = None

    ### Backbone Config ###
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"

    ### Added in to_dict() method so this Config is compatible with Huggingface Trainer!!! ###
    def to_dict(self):
        return asdict(self)

def compute_encoded_length(input_lengths: torch.Tensor, conv_kernels: list, conv_strides: list) -> torch.Tensor:
    """
    Compute the lenght of wav after going through encoders (conv layers)
    Helps to understand distiguish which part of encoded embedding is from padded part of audio and real part of wav

    Args:
        input_lengths (batch_size, audio_length): tensor containing the original length of each audio
        conv_kernels list[num_conv_layers]: list of the conv kernels which audio going to passthrough
        conv_strides list[num_conv_layers]: list of the conv strides which audio going to passthrough

    Returns:
        encoded_length (batch_size, encoded_length): tensor containing the encoded length or in another word, length after wav pass through the encoder cov layers
    """
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths)

    def _compute_conv_out(length, kernel, stride):
        """Compute single conv out"""
        return torch.floor((length - (kernel-1) - 1) / stride) + 1

    for kernel, stride in zip(conv_kernels, conv_strides):
        input_lengths = _compute_conv_out(input_lengths, kernel, stride)
    
    input_lengths = input_lengths.type(torch.int)

    return input_lengths


def compute_sub_attention_mask(config, attention_mask):
    """
    Compute mask for encoded features (wav -> feature_encoder -> encoded_features), 
    for masking the features (in the encoder output, not in the raw wav) that are
    generated from padded part of wav data. 
    encoded_lengths will help you to distiguish which part of encoded features are generated
    from the real part of wav data and which part are generated from padded part of wav data.
    
    Args:
        config (Wav2Vec2Config): model configuration
        attention_mask (torch.Tensor): tensor of shape (batch_size, audio_length) containing the mask for raw wav data

    Returns:
        sub_attention_mask (torch.Tensor): mask/boolean tensor of shape (batch_size, max_encoded_features) for encoded features telling which features in a batch are valid or not valid
    """
    batch_size = attention_mask.shape[0]
    raw_lengths = attention_mask.sum(dim=1) # (batch, wav_lenght)
    # encoded_feature_length / sequence_length if there were no padding on wav
    encoded_lengths = compute_encoded_length(raw_lengths, config.conv_kernels, config.conv_strides)
    # mask for encoded features that generated from padding part of the wav
    # max(encoded_lengths) is the max length of the encoded features among the batch or sequence length
    sub_attention_mask = torch.zeros(size=(batch_size, max(encoded_lengths)))
    
    # this mask tells which embeddings/features in a batch are real or from padded region
    for idx, lenght in enumerate(encoded_lengths):
        sub_attention_mask[idx, :int(lenght)] = 1
    return sub_attention_mask


def compute_span_masking(
    shape,
    mask_prob=0.065,
    mask_length=10,
    min_masks=2,
    sub_attention_mask=None,
    p_replace=0.8 
    ):
    """Main pretraining mask, mask some audio encoded features and it's adjecent n number of features
       which then used to predict this masked tokens with all sarrounding features like we do in bert,
       we mask random tokens to make model predict using all other tokens which is the context.

       This will only have 1's in span masked position, nothing to do with other padded masking.
       
       Args: 
            shape: shape of the audio encoded features to get batch and sequence length
            mask_lenght: num of adjecent features of selected feature to mask (audio features very correlated and overlapped 
                         so masking and then predicting one feature at a timestep will be too easy, since two adjecent feature
                         may not have have much change or more information like word token have, it's importent to mask
                         some n number of adjecent featuresz)
            min_masks: 
            sub_attention_mask:
            p_replace:
            
    """
    batch_size, max_features_in_batch = shape

    if sub_attention_mask is not None:
        sequence_lengths = sub_attention_mask.sum(dim=1).to(torch.int).tolist()
    else:
        sequence_lengths = [max_features_in_batch] * batch_size
    
    all_span_mask = []
    for length in sequence_lengths:

        # mask with max feature len in batch
        span_mask = torch.zeros(max_features_in_batch)

        # select all values that less than mask_prob, rand generate number between 0 to 1 in uniform distribution
        # nonzero tells all the idx where nonzero exist
        mask_idx = (torch.rand(length) <= mask_prob).nonzero()

        # we need to mask the 10 number adjecent numbers of selected mask idx
        span_range = torch.arange(mask_length)
        mask = (mask_idx + span_range).flatten()

        # while adding mask to adjecent, there is a chance the number
        # will go over than sequence length so trim it
        mask = mask[mask <= length - 1]
        span_mask[mask] = 1
        all_span_mask.append(span_mask)
    
    return torch.stack(all_span_mask, dim=0)

# TODO: Some problem with Sample negatives
def sample_negative_indices(feature_shape, num_negatives, mask_time_indices):

    batch_size, max_features_len = feature_shape

    sample_negatives = torch.zeros(
        (batch_size, max_features_len, num_negatives),
        dtype=torch.long
    )

    # positive samples will be the samples in it's correct position, 
    # for eg: 3rd vector positve sample will be 3rd vector itself, 
    # so we use arange that give each vector it's index
    # expand makes each index repeats in num negative
    positive_samples = (
        torch.arange(max_features_len)
        .unsqueeze(1)
        .expand(max_features_len, num_negatives)
    )


    for batch in range(batch_size):

        masked_idx = mask_time_indices[batch].nonzero(as_tuple=True)[0]

        num_masked = len(masked_idx)

        indices_sample = torch.randint(
            low=0,
            high=num_masked - 1,
            size=(num_masked, num_negatives)
        )

        negative_sampled = masked_idx[indices_sample]

        # Avoid selecting the positive feature
        positive_mask = (
            positive_samples[masked_idx] == negative_sampled
        )

        negative_sampled[positive_mask] += 1

        sample_negatives[batch][masked_idx] = negative_sampled

        # IMPORTANT: account for batch when flattening
        sample_negatives[batch] += batch * max_features_len

    return sample_negatives.reshape(
        batch_size * max_features_len,
        num_negatives
    )


def get_audio_duration(audio_path: str):
    "Return Audio duration in Seconds"
    audio_info = sf.info(audio_path)
    return audio_info.frames/audio_info.samplerate



if __name__ == "__main__":
    sampling_length = [10000, 15000]
    data = [torch.rand(l) for l in sampling_length]
    attention_mask = [torch.ones(l) for l in sampling_length]

    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True) # shape: (2, 15000)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True) # shape: (2, 15000)
    
    config = Wav2Vec2Config()
    encoded_length = compute_encoded_length(sampling_length, config.conv_kernels, config.conv_strides)
    sub_attention_mask = compute_sub_attention_mask(config, attention_mask)

    span_mask = compute_span_masking(
        shape=sub_attention_mask.shape,
        sub_attention_mask=sub_attention_mask
    )

    negative_idx = sample_negative_indices(
        feature_shape=sub_attention_mask.shape, 
        num_negatives=5, 
        mask_time_indices=span_mask
    )

    print(negative_idx)



   

