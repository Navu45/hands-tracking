import copy
from typing import Optional

import torch
from torch import nn, Tensor

from core.ops import build_act_layer, SeparableConv2d


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ConvolutionalProjection(nn.Module):
    def __init__(self,
                 in_channels, out_channels, norm_type):
        super().__init__()
        self.conv_proj = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, padding=1, norm_type=norm_type),
            nn.Flatten(start_dim=2),
        )

    def forward(self, x):
        return self.conv_proj(x)


class SeparableConvMHA(nn.Module):
    def __init__(self,
                 hidden_dim,
                 norm_type='BN',
                 value_proj_act_type='ReLU'):
        super().__init__()
        # L-projection of latent vector
        self.latent_proj = nn.Sequential(
            ConvolutionalProjection(hidden_dim, 1, norm_type),
            nn.Softmax(dim=2)
        )

        # keys projection
        self.keys_proj = ConvolutionalProjection(hidden_dim, hidden_dim, norm_type)

        # values projection
        self.values_proj = nn.Sequential(
            ConvolutionalProjection(hidden_dim, hidden_dim, norm_type),
            build_act_layer(value_proj_act_type)
        )

        self.pw_conv = nn.Conv2d(hidden_dim, hidden_dim,
                                 kernel_size=1)

    # @torch.jit.script
    def forward(self, x):
        """
        Applies separable self-attention to input x
        Args:
            x (torch.Tensor): tensor of size N, C, H, W
        Returns:
            y (torch.Tensor): tensor of the same shape as x
        """
        # L, K, V -> (N, 1, K), (N, D, K), (N, D, K) -> D - channels or hidden dim; K == HW; N - batch size
        latent, keys, values = self.latent_proj(x), self.keys_proj(x), self.values_proj(x)
        context = torch.einsum('nik,ndj->nid', latent, keys)  # (N, D, 1)
        context_aware_values = torch.einsum('nd,ndk->ndk', context, values)
        return self.pw_conv(context_aware_values.reshape(x.shape))


class ConvTransformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 n_heads=8,
                 depth=3,
                 mlp_ratio=0.2,
                 drop_rate=0.25):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        encoder_layer = EncoderLayer(hidden_dim, n_heads, drop_rate, mlp_ratio)
        self.encoder = Encoder(encoder_layer, depth, nn.LayerNorm(hidden_dim))

        decoder_layer = DecoderLayer(hidden_dim, n_heads, drop_rate, mlp_ratio)
        self.decoder = Decoder(decoder_layer, depth, nn.LayerNorm(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                enc_img_features, cam_token,
                joint_tokens, pos_encod,
                extra_img_features=None, attention_mask=None):
        device = enc_img_features.device
        HW, B, _ = enc_img_features.shape
        mask = torch.zeros((B, HW), dtype=torch.bool, device=device)

        # Transformer Encoder
        zero_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)  # batch_size X 1
        memory_mask = torch.cat([zero_mask, mask], dim=1)  # batch_size X (1 + height * width)
        cam_with_img = torch.cat([cam_token, enc_img_features], dim=0)  # (1 + h * w) X batch_size X feature_dim
        e_outputs = self.encoder(cam_with_img,
                                 key_padding_mask=memory_mask,
                                 pos_encod=pos_encod,
                                 extra_img_features=extra_img_features)  # (1 + h * w) X batch_size X feature_dim
        cam_features, enc_img_features = e_outputs.split([1, HW], dim=0)

        # Transformer Decoder
        zero_tgt = torch.zeros_like(joint_tokens)  # num_j X batch_size X feature_dim
        joint_features = self.decoder(joint_tokens, enc_img_features, target_mask=attention_mask,
                                      memory_key_padding_mask=mask, pos_encod=pos_encod,
                                      query_pos=zero_tgt)  # num_j X batch_size X feature_dim

        return cam_features, enc_img_features, joint_features


class Decoder(nn.Module):
    def __init__(self, decoder_layer, depth, norm):
        super().__init__()
        self.layers = clone(decoder_layer, depth)
        self.norm = norm

    def forward(self, target, memory, **kwargs):
        out = target
        for layer in self.layers:
            out = layer(out, memory, **kwargs)
        return self.norm(out)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, depth, norm):
        super().__init__()
        self.layers = clone(encoder_layer, depth)
        self.norm = norm

    def forward(self, source, **kwargs):
        out = source
        for layer in self.layers:
            out = layer(out, **kwargs)
        return self.norm(out)


class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_heads,
                 drop_rate,
                 mlp_ratio):
        super().__init__()
        # Self-attention sublayer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, drop_rate)
        self.dropout1 = nn.Dropout(drop_rate)
        # MLP sublayer
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(mlp_ratio * embed_dim), drop_rate=drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

    @staticmethod
    def with_pos_encoding(x, pos):
        return x if pos is None else torch.cat([x[:1], (x[1:] + pos)], dim=0)

    def forward(self,
                source, pos_encod=None,
                key_padding_mask=None, attn_mask=None,
                extra_img_features=None):
        # Self-attention sublayer
        out = self.norm1(source)
        queries = keys = self.with_pos_encoding(out, pos_encod)
        out = self.attention(queries, keys, value=out,
                             key_padding_mask=key_padding_mask,
                             attn_mask=attn_mask)[0]
        # MLP sublayer
        residual = source + extra_img_features + self.dropout1(out)
        out = self.norm2(out)
        out = self.mlp(out)
        return residual + self.dropout2(out)


class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_heads,
                 drop_rate,
                 mlp_ratio):
        super().__init__()
        # Self-attention sublayer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads,
                                                    drop_rate, batch_first=True)
        self.dropout1 = nn.Dropout(drop_rate)

        # Cross-attention sublayer
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, n_heads, drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        # MLP sublayer
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(mlp_ratio * embed_dim), drop_rate=drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)

    @staticmethod
    def with_pos_encoding(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_encod: Optional[Tensor] = None):
        # Self-attention sublayer
        out = self.norm1(target).permute(1, 2, 0)
        out = self.self_attention(query=out,
                                  key=out,
                                  value=out,
                                  key_padding_mask=key_padding_mask,
                                  attn_mask=target_mask)[0].permute(2, 0, 1)

        # Cross-attention sublayer
        resid = target + self.dropout1(out)
        out = self.norm2(resid)
        out = self.cross_attention(query=out,
                                   key=self.with_pos_encoding(memory, pos_encod),
                                   value=memory,
                                   key_padding_mask=memory_key_padding_mask,
                                   attn_mask=memory_mask)[0]
        # MLP sublayer
        resid = resid + self.dropout2(out)
        out = self.norm3(resid)
        out = self.mlp(out)
        return resid + self.dropout3(out)


class MLP(nn.Module):
    def __init__(self,
                 in_ch: int,
                 hidden_features=None,
                 drop_rate=0.):
        super().__init__()
        hidden_features = hidden_features or in_ch
        self.layer = nn.Sequential(
            nn.Linear(in_ch, hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity(),
            nn.Linear(hidden_features, in_ch)
        )

    def forward(self, x):
        return self.layer(x)
