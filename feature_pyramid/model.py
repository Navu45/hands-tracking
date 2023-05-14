from collections import namedtuple

import pytorch_lightning
import torch
from torch import nn

from core.ops import SeparableConv2d
from feature_pyramid.transformer import ConvTransformer


class MultiscaleFusion(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.downsampling = nn.ModuleList([
            SeparableConv2d(in_channels_list[i - 1], in_channels_list[i - 1], 3, 2, 1,
                            norm_type='BN')
            for i in range(1, len(in_channels_list))
        ])

    def forward(self, multiscale_x: tuple[torch.Tensor]):
        output = [multiscale_x[0]]
        for i, downsampling in enumerate(self.downsampling):
            output.append(torch.cat([
                multiscale_x[i + 1],
                downsampling(multiscale_x[i])
            ], dim=1))
        return output


class MultiScaleFusionTransformerLayer(nn.Module):  # Queen fusion
    def __init__(self,
                 in_channels_list, fused_channels_list, out_channels_list,
                 depths, mlp_ratios,
                 path='down',
                 transformer_norm_type='BN',
                 mlp_drop_rate=0.,
                 mlp_act_type='GELU',
                 attn_proj_act_type='ReLU',
                 attn_norm_type='BN',
                 drop_path_rate=0.):
        super().__init__()
        self.use_down_path = path == 'down'
        self.multiscale_fusion = MultiscaleFusion(in_channels_list)

        if self.use_down_path:
            updown_path = [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=2, padding=1,
                          groups=out_channels)
                for out_channels in (out_channels_list[1:]
                                     if not self.use_down_path else
                                     out_channels_list[:-1])
            ]
            self.updown_path = nn.ModuleList([nn.Identity()] + updown_path)
        else:
            self.updown_path = nn.ModuleList([
                nn.UpsamplingNearest2d(scale_factor=2)
                for _ in range(len(out_channels_list))
            ])

        transformers = [
            ConvTransformer(in_channels, out_channels,
                            depth=depth,
                            transformer_norm_type=transformer_norm_type,
                            mlp_ratio=mlp_ratio,
                            mlp_drop_rate=mlp_drop_rate,
                            mlp_act_type=mlp_act_type,
                            attn_proj_act_type=attn_proj_act_type,
                            attn_norm_type=attn_norm_type,
                            drop_path_rate=drop_path_rate)
            if in_channels != -1 else nn.Identity()
            for in_channels, out_channels, depth, mlp_ratio in zip(fused_channels_list,
                                                                   out_channels_list,
                                                                   depths,
                                                                   mlp_ratios)
        ]
        if not self.use_down_path:
            transformers.reverse()
        self.transformers = nn.ModuleList(transformers)

    def forward(self, multiscale_x: tuple[torch.Tensor]):
        multiscale_x = self.multiscale_fusion(multiscale_x)
        output = []
        if not self.use_down_path:
            multiscale_x.reverse()
        for i, (fusion_block, updown_path) in enumerate(zip(self.transformers, self.updown_path)):
            x = [multiscale_x[i]]
            if len(output) != 0:
                x.append(updown_path(output[-1]))
            x = torch.cat(x, dim=1)
            output.append(fusion_block(x))
        if not self.use_down_path:
            output.reverse()
        return output


class ChannelLast(nn.Module):
    @staticmethod
    def forward(x):
        return x.permute(0, 2, 1)


class ZeroHead(nn.Module):
    def __init__(self, in_channels_list, avg_pool_outputs, num_joints, num_classes):
        super().__init__()
        self.tasks = [task for i, (task, out_features) in enumerate(zip(['class', 'keypoints'],
                                                                        [num_classes, num_joints]))
                      if out_features != 0]
        self.output = {
            task: namedtuple('task', [task])
            for task in self.tasks
        }
        self.heads = nn.ModuleList([
            nn.ModuleDict(
                {
                    task: nn.Sequential(
                        nn.AdaptiveAvgPool2d((avg_pool_out, 2)),
                        nn.Flatten(start_dim=1, end_dim=-2),
                        ChannelLast(),
                        nn.Linear(avg_pool_out * out_features, num_features),
                        ChannelLast()
                    )
                    for i, (task, num_features) in enumerate(zip(['class', 'keypoints'],
                                                                 [num_classes, num_joints])) if num_features != 0
                }
            )
            for out_features, avg_pool_out in zip(in_channels_list, avg_pool_outputs)
        ])

    def forward(self, multifusion_x: list[torch.Tensor]):
        output = ()
        for i, multi_head in enumerate(self.heads):
            for task, head in multi_head.items():
                if task in multi_head:
                    output += self.output[task](head(multifusion_x[i]).unsqueeze(-1))
        return torch.mean(torch.cat(output, dim=-1), dim=-1)


class TransformerFCN(pytorch_lightning.LightningModule):
    def __init__(self,
                 in_channels_layers: list[list[int]],
                 fused_channels_layers: list[list[int]],
                 out_channels_layers: list[list[int]],
                 depths_layers: list[list[int]],
                 mlp_ratio_layers: list[list[int]],
                 transformer_norm_type='BN',
                 mlp_drop_rate=0.,
                 mlp_act_type='GELU',
                 attn_proj_act_type='ReLU',
                 attn_norm_type='BN',
                 drop_path_rate=0.,
                 avg_pool_outputs=None,
                 num_joints=21,
                 num_classes=None):
        super().__init__()
        # build transformers
        self.fusion_layers = nn.Sequential(*[
            MultiScaleFusionTransformerLayer(
                in_channels_list, fused_channels_list, out_channels_list,
                depths, mlp_ratios,
                path='up' if i % 2 == 0 else 'down',
                transformer_norm_type=transformer_norm_type,
                mlp_drop_rate=mlp_drop_rate,
                mlp_act_type=mlp_act_type,
                attn_proj_act_type=attn_proj_act_type,
                attn_norm_type=attn_norm_type,
                drop_path_rate=drop_path_rate
            )
            for i, (in_channels_list, fused_channels_list, out_channels_list, depths, mlp_ratios)
            in enumerate(zip(in_channels_layers,
                             fused_channels_layers,
                             out_channels_layers,
                             depths_layers, mlp_ratio_layers))
        ])
        self.zero_head = ZeroHead(out_channels_layers[-1],
                                  avg_pool_outputs,
                                  num_joints,
                                  num_classes)

    def forward(self, multiscale_img_features):
        fcn_outputs = self.fusion_layers(multiscale_img_features)
        outputs = self.zero_head(fcn_outputs)
        return outputs
