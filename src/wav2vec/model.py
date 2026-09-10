import os
from typing import Any
import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils import Wav2Vec2Config



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
            probs = probs[mask.flatten()]
            
        print(probs.shape)
        
    def forward(self, hidden_state, span_mask=None):
        batch_size, seq_len, hidden_dim = hidden_state.shape
        
        print(f"hidden state before reshaping :{hidden_state.shape}")
        
        # converting each hidden state to a feature with number of codevector as dim, 
        # so we can take argmax value from it to choose which codevector you are going to use.
        hidden_state = self.weight_proj(hidden_state).reshape(batch_size * seq_len * self.num_codebooks, -1)
        print(f"hidden state after linear layer and reshaping: {hidden_state.shape}")
        
        if self.training:
            # Softmax but differentiable way
            # Setting True will make small value to zero and argmax value as 1
            codevector_prob = F.gumbel_softmax(hidden_state.float(), tau=self.temperature, hard=True)
            
            # Compute perplexity which we need as a loss for making the codevector choosing even
            hidden_state = hidden_state.reshape(batch_size * seq_len, self.num_codebooks, -1)
            codevector_soft_dis = hidden_state.softmax(axis=-1)
            self._compute_perplexity(codevector_soft_dis, span_mask)
            
        
        

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
        return normed_x, projected_x
    
        
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
        
    
    def forward(self, x, attention_mask= None):
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
        
        
        extracted_features = self.feature_extraction_layer(input_values.unsqueeze(1)).transpose(1, 2)
        normed_x, projected_x = self.projection_layer(extracted_features)
        encoder_output = self.encoder_layer(projected_x)
        
        if return_features_to_quantize:
            return encoder_output, extracted_features
        
        else:
            return encoder_output


class Wav2Vec2ForPretraining(nn.Module):
    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_layer = nn.Dropout(config.pre_quantizer_dropout)
        self.quantizer = Wav2Vec2GumbleVectorQuantizer(config)
    
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
        
        print(f"Transformer Output :{transformer_output.shape}")
        print("Features to quantize", features_to_quantize.shape)
        a = self.quantizer(features_to_quantize, mask_time_indices)
        
        
        

        
        
        
    
if __name__ == "__main__":
    from wav2vec.dataset import LibriSpeechDataset
    from torch.utils.data import DataLoader
    from wav2vec.dataset import Wav2Vec2CollateFunctionForPretraining
    
    config = Wav2Vec2Config()
    
    dataset = LibriSpeechDataset(include_splits="dev")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Wav2Vec2CollateFunctionForPretraining(config))
    
    wav2vec_pretrain = Wav2Vec2ForPretraining(config)
    data_iter = iter(dataloader)
    next(data_iter)
    input_data = next(data_iter)
    
    x = wav2vec_pretrain(**input_data)