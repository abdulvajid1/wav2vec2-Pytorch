import os
from typing import Any
import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import Wav2Vec2Config, get_logger

from .utils import Wav2Vec2ForPreTrainingOutput

logger = get_logger("wav2vec2.model")


class Wav2Vec2GumbleVectorQuantizer(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        
        self.num_codebooks = config.num_codevector_groups
        self.num_codes = config.num_codevectors_per_group
        
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_codebooks * self.num_codes, config.codevector_dim//self.num_codebooks)
            )
        
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_codebooks * self.num_codes)
        self.temperature = 2
    
    def _compute_perplexity(self, probs, mask=None):

        if mask is not None:
            # only taking perplexity loss on masked positions codevectores, since it's the only vectors influence the loss.
            # each masked vector will have codevector which tells, what is probabilty of this vector to choose a code vector
            # from all the codevectores, we only need make that codevector uniform distribution. so take all codeprob of 
            # span masked vectors and calculate perplexity on them.
            
            marginal_probs = probs[mask.flatten()]
            # next we have prob vector for each span masked vector, we will calculate mean on dim=0 which tell
            # what is the average chance of the each codevector is choosen, which give [0.1, 0.2, 0.2, 0.3, 0.2]
            # which tell average of each codevector to be choosen, now we calculate the perplexity to make this averged vector uniform 

        else:
            marginal_probs = probs.mean(0)


        # logic is: log(prob) is the NLL loss or binary entropy loss, to make it maximum, 
        # each mean marginal prob should small as possible, since that how log work, to make it to
        # single number for loss, we do summation on all nll loss and exp that to make it perplexity
        perplexity = torch.exp(- torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1))

        # we still have num codebooks, we sum it to get single number loss 
        return perplexity.sum()

            
        
    def forward(self, hidden_state, span_mask=None):
        batch_size, seq_len, hidden_dim = hidden_state.shape
        
        # converting each hidden state to a feature with number of codevector as dim, 
        # so we can take argmax value from it to choose which codevector you are going to use.
        hidden_state = self.weight_proj(hidden_state)
        hidden_state = hidden_state.reshape(batch_size * seq_len * self.num_codebooks, -1)
        
        
        if self.training:
            # Softmax but differentiable way
            # Setting True will make small value to zero and argmax value as 1
            codevector_prob = F.gumbel_softmax(hidden_state.float(), tau=self.temperature, hard=True)
            
            # Compute perplexity which we need as a loss for making the codevector choosing even
            hidden_state = hidden_state.reshape(batch_size * seq_len, self.num_codebooks, -1)
            codevector_soft_dis = hidden_state.softmax(axis=-1)
            perplexity = self._compute_perplexity(codevector_soft_dis, span_mask)

        else:

            # get index of max value in each hidden state
            codevector_prob_idx = hidden_state.argmax(dim=-1) 

            # create placement & make all one in hidden state which have max value (chosen quatizer index) 
            # & make choosen hidden state full of onees and other zero so we can multiply it with qutizer codebook\
            # so we get the correct quatnizer vectore
            codevector_prob = torch.zeros_like(hidden_state, device=hidden_state.device)
            codevector_prob[torch.arange(hidden_state.shape[0]), codevector_prob_idx] = 1

            codevector_prob.reshape(batch_size * seq_len, self.num_codebooks, -1)
            perplexity = self._compute_perplexity(codevector_prob, mask=span_mask)


        # Since we choose two codebooks for every sameple, we need to concate that
        # we choose 2 codebooks cuz, managing one big codebook is hard (need to know why), 
        # so better method is using multiple small codebooks

        # concate two chosen code vector to 1 for multiplication
        codevector_prob = codevector_prob.reshape(batch_size * seq_len, -1) # [0 , 0, 1], [0,1,0] -> [0, 0, 1, 0, 1, 0]
        codevector_per_group = codevector_prob.unsqueeze(-1) * self.codevectors.type_as(codevector_prob) # choosing only codevectores and making other zero
        codevector = codevector_per_group.reshape(batch_size*seq_len, self.num_codebooks, self.num_codes, -1)

        # now sum to remove all zero vectors and keep only quantizer vector
        codevector = codevector.sum(dim=-2)
        codevector = codevector.reshape(batch_size, seq_len, -1) # concatenate the chose quantizers

        return codevector, perplexity 


        
class Wav2Vec2NormConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias
    ) -> None:
        
        super().__init__() 
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.norm_layer = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    
    def forward(self, x: torch.Tensor):
        # shape: (batch, channels, length)
        x = self.conv_layer(x)
        x = x.transpose(dim0=-1, dim1=-2)
        x = self.norm_layer(x)
        x = x.transpose(dim0=-2, dim1=-1)
        x = self.activation(x)
        return x



class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        
        assert len(config.conv_dim) == len(config.conv_kernels) == len(config.conv_strides), "number of convolution layers, kernels, strides didn't match up"
        num_conv_layers = len(config.conv_dim)
        conv_channels = (1,) + tuple(config.conv_dim) # this add initial channel of our audio, our initial channel will be 1
        
        self.conv_layers = nn.ModuleList()
        for conv_idx in range(num_conv_layers):
            self.conv_layers.append(
                Wav2Vec2NormConvLayer(
                    in_channels=conv_channels[conv_idx], # we use this here cuz conv_channels have 1 at 0 index but other don't, so indexing will be coorect
                    out_channels=self.config.conv_dim[conv_idx],
                    kernel_size=self.config.conv_kernels[conv_idx],
                    stride=self.config.conv_strides[conv_idx],
                    bias=config.conv_bias                    
                )
            )
            
    def forward(self, x: torch.Tensor):
        for layer in self.conv_layers:
            x = layer(x)
        return x



class Wav2Vec2Attention(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dimension // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        
        attention_out = F.scaled_dot_product_attention(q, k, v, 
                                                       attn_mask=attention_mask, 
                                                       dropout_p=self.config.attention_dropout_p)
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)
        
        return attention_out



class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        hidden_size = config.embedding_dimension * config.mlp_ratio
        self.intermediate_layer = nn.Linear(config.embedding_dimension, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.mlp_dropout_p)
        self.output_layer = nn.Linear(hidden_size, config.embedding_dimension)
    
    def forward(self, x):
        x = self.intermediate_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        x = self.dropout(x)
        return x
    
class Wav2Vec2EncoderBlock(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.attn = Wav2Vec2Attention(config)
        self.dropout = nn.Dropout(config.transformer_encoder_dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attn(x, attention_mask))
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)
        return x
    
    
    
class Wav2Vec2PositionalEncoding(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(
            in_channels=config.embedding_dimension,
            out_channels=config.embedding_dimension,
            kernel_size=config.conv_positional_emb_kernel_size,
            padding=config.conv_positional_emb_kernel_size//2, # doing this will make no changes in shapes
            groups=config.conv_positional_emb_groups
        )
        self.activation = nn.GELU()
    
    def forward(self, x):
        batch_size, seq_len, n_embedding = x.shape
        # shape: (batch, seq, embedding) -> (batch, emb, seq)
        # we slide through seq dim
        x = x.transpose(1, 2)
        positional_embedding = self.conv(x)
        positional_embedding = positional_embedding[:, :, :seq_len]
        positional_embedding = self.activation(positional_embedding) 
        return positional_embedding.transpose(1, 2)



class Wav2Vec2ProjectionLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__() 
        self.projection_layer = nn.Linear(in_features=config.conv_dim[-1], out_features=config.embedding_dimension)
        self.norm_layer = nn.LayerNorm(config.conv_dim[-1])
        self.dropout_layer = nn.Dropout(config.feature_projection_dropout_p)
    
    def forward(self, x):
        normed_x = self.norm_layer(x) # we need normed x for constrastiv loss, and projected x for transformer input
        projected_x = self.projection_layer(normed_x)
        projected_x = self.dropout_layer(projected_x)
        return projected_x, normed_x



class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.config = config
        
        self.pos_conv_embed = Wav2Vec2PositionalEncoding(config)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.dropout = nn.Dropout(config.conv_positional_emb_drop_p)
        
        self.encoder_blocks = nn.ModuleList([
            Wav2Vec2EncoderBlock(config) for _ in range(config.num_transformer_layers)
        ])
        
    
    def forward(self, x, attention_mask=None):

        batch_size, seq_len, dim = x.shape
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
            x[~attention_mask] = 0 # here we are setting the masked features into zero since they have values due to convolution operatoin with mix real value and padded value
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1) # flash attention expect output like this format
            
        positional_embedding = self.pos_conv_embed(x) #
        x += positional_embedding
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x, attention_mask)

        return x
        
        

class Wav2Vec2Model(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.feature_extraction_layer = Wav2Vec2FeatureExtractor(config)
        self.projection_layer = Wav2Vec2ProjectionLayer(config)
        self.encoder_layer = Wav2Vec2Encoder(config)
    
    def forward(self, 
                input_values,
                attention_mask=None,
                sub_attention_mask=None,
                span_mask=None,
                return_features_to_quantize=False):

        logger.info(f"Raw input shapes: {input_values.shape}")
        extracted_features = self.feature_extraction_layer(input_values.unsqueeze(1)).transpose(1, 2)
        hidden_state , extract_features = self.projection_layer(extracted_features)
        encoder_output = self.encoder_layer(hidden_state)
        
        if return_features_to_quantize:
            return encoder_output, extract_features
        
        else:
            return encoder_output



class Wav2Vec2ForPretraining(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_layer = nn.Dropout(config.pre_quantizer_dropout)
        self.quantizer = Wav2Vec2GumbleVectorQuantizer(config)

        self.proj_transformer = nn.Linear(config.embedding_dimension, config.codevector_dim)
        self.proj_codevector = nn.Linear(config.codevector_dim, config.codevector_dim)

    def cosine_similarity(self, target_features, negative_features, predicted_features, temperature=0.1):
        # target feature is the postive quantized codevector, negative features are the
        # negative quantized codevectors, predicted features are the transformer output
        # projected to codevector dim, we concatenate the target/positive and negative features to 
        # make a single tensor of shape (1 + num_negatives, batch_size, seq_len, feature_dim)

        target_features = target_features.unsqueeze(0)  # shape: (1, batch_size, seq_len, feature_dim) 
        target_features = torch.cat([target_features, negative_features], dim=0)  # shape: (1 + num_negatives, batch_size, seq_len, feature_dim)

        # Now we calculate the cosine similarity between the predicted features and the concatenated
        # target/positive and negative features

        # cosine similarity between (batch_size, seq_len, feature_dim) and (positive_target + num_negatives, batch_size, seq_len, feature_dim)
        cosine_sim = torch.cosine_similarity(predicted_features, target_features, dim=-1) / temperature
        return cosine_sim


    def forward(self, 
                input_values,
                attention_mask=None,
                sub_attention_mask=None,
                mask_time_indices=None,
                sampled_negatives=None,
                return_features_to_quantize=False):
        
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        transformer_output, features_to_quantize = self.wav2vec2(input_values, 
                                                                 attention_mask, 
                                                                 sub_attention_mask, 
                                                                 mask_time_indices, 
                                                                 return_features_to_quantize=True)
        
        logger.info(f"Transformer Output: {transformer_output.shape}")
        logger.info(f"Features to quantize: {features_to_quantize.shape}")

        if mask_time_indices is not None:
            logger.info(f"mask_time_indices: {mask_time_indices.shape}")

        codevectors, perplexity = self.quantizer(features_to_quantize, mask_time_indices)

        logger.info(f"Codevector {codevectors.shape}")
        logger.info(f"perplexity {perplexity}")

        # Project down the tranformer output for contrastive loss with quantized codebook vector
        quantized_vectors = self.proj_codevector(codevectors)
        transformer_output = self.proj_transformer(transformer_output)
        logger.info(f"Quantized Vectors: {quantized_vectors.shape}")
        logger.info(f"Transformer Output: {transformer_output.shape}")

        loss = None
        diversity_loss = None
        contrastive_loss = None

        if sampled_negatives is not None:
            batch_size, seq_len, vq_size = quantized_vectors.shape
            _, num_negatives = sampled_negatives.shape


            logger.info(f"Quantized Vectors: {quantized_vectors.shape}")
            logger.info(f"Sampled Negatives: {sampled_negatives.shape}")
            logger.info(f"sampled_negatives: {sampled_negatives}")
            negative_quantized_codes = quantized_vectors.reshape(-1, vq_size)[sampled_negatives.flatten()]

            logger.info(f"Negative Quantized Codes: {negative_quantized_codes.shape}")


            # Reshape negative quantized codes to (num_negatives, batch_size, seq_len, vq_size)
            # So each will be audio feature (batch, seq, vq_size) will be negative samples making (num_negatives, batch, seq, vq_size)

            negative_quantized_codes = negative_quantized_codes.reshape(batch_size, seq_len, num_negatives, vq_size).permute(2, 0, 1, 3) # shape: (num_negatives, batch_size, seq_len, vq_size)
            logger.info(f"Negative Quantized Codes After Permute: {negative_quantized_codes.shape}")




            cosine_similarity = self.cosine_similarity(quantized_vectors, 
                                                       negative_quantized_codes,
                                                       transformer_output,
                                                       temperature=self.config.contrastive_logits_temperature)

            logger.info(f"Cosine Similarity: {cosine_similarity.shape}")


            # the intution here, it's true that negative and positive samples are different when we creat negative samples,
            # but when each of them pick a quantizer vector, there is high chance at initial time of the training, 
            # the negative sample and positive sample will pick same quantizer vector, which will make the cosine similarity to be 1, 
            # which will make the softmax of that negative sample to be 1, which will make the loss to be NaN, 
            # so we need to check if any negative sample is equal to positive sample and if yes, we need to set 
            # the similarity of that negative sample to be very low value, so it won't contribute to the loss.
            neg_equals_pos_mask = (quantized_vectors == negative_quantized_codes).all(dim=-1)

            if neg_equals_pos_mask.any():
                logger.warning(f"Some negative samples are equal to positive samples. This may lead to NaN loss values. Please check your negative sampling strategy.")

                # Set the similarity of negative samples that are equal to positive samples to a very low value,
                # so they don't contribute to the loss due to how cross entropy loss works,
                # it will make the softmax of that negative sample to be zero, so it won't contribute to the loss
                cosine_similarity[1: ][neg_equals_pos_mask] = float('-inf')  


            cosine_similarity = cosine_similarity.permute(1, 2, 0) # shape: (batch_size, seq_len, 1 + num_negatives)
            cosine_similarity = cosine_similarity.reshape(batch_size * seq_len, cosine_similarity.shape[-1]) # shape: (batch_size * seq_len, 1 + num_negatives)

            # we create labels, we keep all -100 except the masked positions, we only need loss on the masked positions
            # the first label will be the positive sample, so we set it as 1 and all other labels as 0's
            labels = torch.ones(len(cosine_similarity), dtype=torch.long, device=cosine_similarity.device) * -100

            # now for each negatives, we set label to 0 , since we know first vector is the positive sample
            labels[mask_time_indices.flatten()] = 0
            contrastive_loss = F.cross_entropy(cosine_similarity, labels, reduction="sum") / mask_time_indices.sum()

            GV = self.config.num_codevector_groups * self.config.num_codevectors_per_group
            diversity_loss = ((GV - perplexity) / GV) * mask_time_indices.sum()

            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

            return Wav2Vec2ForPreTrainingOutput(
                loss=loss,
                projected_features=transformer_output,
                quantized_features=quantized_vectors,
                codevector_perplexity=perplexity,
                contrastive_loss=contrastive_loss,
                diversity_loss=diversity_loss
            )


            

        
        
        
    
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    from wav2vec.dataset import LibriSpeechDataset
    from torch.utils.data import DataLoader
    from wav2vec.dataset import Wav2Vec2CollateFunctionForPretraining
    
    config = Wav2Vec2Config()

    
    dataset = LibriSpeechDataset(include_splits="dev-clean", max_audio_duration=5.0)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Wav2Vec2CollateFunctionForPretraining(config))
    
    wav2vec_pretrain = Wav2Vec2ForPretraining(config).to(DEVICE)
    data_iter = iter(dataloader)
    next(data_iter)
    input_data = next(data_iter)

    input_data = {k: v.to(DEVICE) for k, v in input_data.items()}
    
    x = wav2vec_pretrain(**input_data)