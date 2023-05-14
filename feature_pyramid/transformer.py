import torch
from timm.models.layers import DropPath
from torch import nn

from core.ops import build_act_layer, CSPBlock, build_norm_layer, ConvolutionalMLP


class SeparableConvolutionalAttention(nn.Module):  # Separable convolutional self-attention
    def __init__(self,
                 hidden_dim,
                 norm_type='BN',
                 value_proj_act_type='ReLU'):
        super().__init__()

        self.base_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            build_norm_layer(hidden_dim, norm_type)
        )

        # L-projection of latent vector
        self.latent_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # keys projection
        self.keys_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # values projection
        self.values_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            build_act_layer(value_proj_act_type),
        )

        self.pw_conv = nn.Conv2d(hidden_dim, hidden_dim,
                                 kernel_size=1)

    def proj_qkv(self, x):
        x = self.base_conv(x)
        latent = self.latent_proj(x)
        latent = torch.softmax(torch.flatten(latent, start_dim=2), dim=2).view(latent.shape)
        return latent, self.keys_proj(x), self.values_proj(x)

    def forward(self, x):
        """
        Applies separable self-attention to input x
        Args:
            x (torch.Tensor): image features
        Returns:
            y (torch.Tensor): tensor of the same shape as x
        """
        # L, K, V -> (N, 1, H, W), (N, D, H, W), (N, D, H, W)
        latent, keys, values = self.proj_qkv(x)
        context = torch.einsum('nihw,ndhw->nd', latent, keys)  # (N, D, 1, 1)
        context_aware_values = torch.einsum('nd,ndhw->ndhw', context, values)  # (N, D, H, W)
        return self.pw_conv(context_aware_values)


class ConvTransformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim,
                 transformer_norm_type='BN',
                 mlp_ratio=4,
                 mlp_drop_rate=0.,
                 mlp_act_type='GELU',
                 attn_norm_type=None,
                 attn_proj_act_type='ReLU',
                 drop_path_rate=0.1):
        super().__init__()
        self.norm1 = build_norm_layer(hidden_dim, transformer_norm_type)
        self.attn = SeparableConvolutionalAttention(hidden_dim, attn_norm_type, attn_proj_act_type)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(hidden_dim, transformer_norm_type)
        self.mlp = ConvolutionalMLP(hidden_dim, mlp_ratio,
                                    mlp_drop_rate, mlp_act_type)

    def forward(self, img_features):
        residual = self.norm1(img_features)
        x = self.attn(residual)
        x = residual + self.drop_path(x)

        residual = self.norm2(x)
        x = self.mlp(residual)
        return residual + self.drop_path(x)


class ConvTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=3,
                 transformer_norm_type='BN',
                 mlp_ratio=4,
                 mlp_drop_rate=0.1,
                 mlp_act_type='GELU',
                 attn_proj_act_type='ReLU',
                 attn_norm_type='BN',
                 drop_path_rate=0.1):
        super().__init__()
        self.csp_block = CSPBlock(
            in_channels, out_channels,
            nn.Sequential(*[
                ConvTransformerBlock(
                    hidden_dim=out_channels,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    transformer_norm_type=transformer_norm_type,
                    mlp_act_type=mlp_act_type,
                    attn_proj_act_type=attn_proj_act_type,
                    attn_norm_type=attn_norm_type,
                    drop_path_rate=drop_path_rate
                ) for _ in range(depth)])
        )

    def forward(self, enc_img_features):
        return self.csp_block(enc_img_features)
