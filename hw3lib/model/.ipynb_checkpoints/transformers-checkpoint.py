import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding, RotaryPositionalEncoding
from .decoder_layers import CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
from .norms import create_norm
import warnings

'''
TODO: Implement these Modules.

This file contains the transformer architecture used in this homework:

EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
'''

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int, 
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            norm_type: Literal['rmsnorm', 'layernorm'] = 'rmsnorm',
            ffn_type: Literal['swiglu', 'gelu'] = 'swiglu',
            qk_norm: bool = False,
            encoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'] = 'rope',
            decoder_pos_encoding: Literal['rope', 'sinusoidal', 'none'] = 'rope',
    ):
        super().__init__()
        for name, value in [('encoder_pos_encoding', encoder_pos_encoding),
                            ('decoder_pos_encoding', decoder_pos_encoding)]:
            if value not in ['rope', 'sinusoidal', 'none']:
                raise ValueError(f"{name} must be 'rope', 'sinusoidal', or 'none'")

        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_pos_encoding = encoder_pos_encoding
        self.decoder_pos_encoding = decoder_pos_encoding

        encoder_head_dim = d_model // num_encoder_heads
        decoder_head_dim = d_model // num_decoder_heads
        self.encoder_rope = RotaryPositionalEncoding(encoder_head_dim, max_len=max_len) if encoder_pos_encoding == 'rope' else None
        self.decoder_rope = RotaryPositionalEncoding(decoder_head_dim, max_len=max_len) if decoder_pos_encoding == 'rope' else None

        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout
        )
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model=d_model,
                num_heads=num_encoder_heads,
                d_ff=d_ff_encoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                rope=self.encoder_rope,
                qk_norm=qk_norm
            ) for _ in range(num_encoder_layers)
        ])

        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_decoder_heads,
                d_ff=d_ff_decoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                rope=self.decoder_rope,
                qk_norm=qk_norm
            ) for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = create_norm(norm_type, d_model)
        self.decoder_norm = create_norm(norm_type, d_model)
        self.final_linear = nn.Linear(d_model, num_classes)

        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        if self.encoder_pos_encoding == 'sinusoidal':
            x_enc = self.positional_encoding(x_enc)

        x_enc = self.dropout(x_enc)
        pad_mask_src = PadMask(x_enc, x_enc_lengths)

        running_att = {}
        for i, layer in enumerate(self.enc_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
 
            x_enc, attention = layer(x_enc, key_padding_mask=pad_mask_src)
            running_att[f'layer{i+1}_enc_self'] = attention

        x_enc = self.encoder_norm(x_enc)
        ctc_logits = self.ctc_head(x_enc).transpose(0, 1)
        ctc_inputs = {
            'log_probs': ctc_logits,
            'lengths': x_enc_lengths
        }

        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self, 
        padded_targets: torch.Tensor, 
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths)

        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None, unless you are using the decoder as a standalone model or doing inference, you should provide target_lengths")

        causal_mask = CausalMask(padded_targets)

        x_dec = self.target_embedding(padded_targets)
        if self.decoder_pos_encoding == 'sinusoidal':
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
 
            x_dec, self_attn, cross_attn = layer(
                x_dec,
                encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)

        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict, dict]:
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")
        
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(padded_sources, source_lengths)
        seq_out, dec_running_att = self.decode(padded_targets, encoder_output, target_lengths=target_lengths, pad_mask_src=pad_mask_src)
        
        running_att = {**enc_running_att, **dec_running_att}
        
        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor, pad_mask_src: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ValueError("score method is not supported during training")

        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")