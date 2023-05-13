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

    def forward(self, multiscale_x: list[torch.Tensor]):
        output = [multiscale_x[0]]
        for i in range(1, len(multiscale_x)):
            output.append(torch.cat([
                multiscale_x[i],
                self.downsampling[i - 1](multiscale_x[i - 1])
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
            self.updown_path = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=2, padding=1,
                          groups=out_channels)
                for out_channels in (out_channels_list[1:]
                                     if not self.use_down_path else
                                     out_channels_list[:-1])
            ])
        else:
            self.updown_path = nn.UpsamplingNearest2d(scale_factor=2)

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

    def forward_updown_path(self, index, prev_fusion_out):
        if self.use_down_path:
            print(self.updown_path[index], prev_fusion_out.shape)
            return self.updown_path[index](prev_fusion_out)
        return self.updown_path(prev_fusion_out)

    def forward(self, multiscale_x: list[torch.Tensor]):
        multiscale_x = self.multiscale_fusion(multiscale_x)
        output = []
        if not self.use_down_path:
            multiscale_x.reverse()
        for i, (fusion_block, img_features) in enumerate(zip(self.transformers, multiscale_x)):
            x = [img_features]
            if len(output) != 0:
                x.append(self.forward_updown_path(i - 1, output[-1]))
            x = torch.cat(x, dim=1)
            print(x.shape)
            output.append(fusion_block(x))
        return output


# down
# [1 * c, 2 * c, 4 * c],
# [-1, 4 * c, 7 * c],
# [c, c, 2 * c],
# [1, 1, 1],
# [4, 4, 4],

# up
# [2 * c, 4 * c, 8 * c],
# [4 * c, 10 * c, 12 * c],
# [1 * c, 2 * c, 4 * c],
# [1, 1, 1],
# [4, 4, 4],

class Hand3DNet(nn.Module):
    def __init__(self,
                 features_sizes: list,
                 hidden_dims: list,
                 backbone,
                 num_layers=3,
                 num_joints=21,
                 # num_vertices=195,
                 n_heads=8,
                 depth=2,
                 mlp_ratio=2.0,
                 drop_rate=0.1):
        super().__init__()
        self.backbone = backbone
        # self.num_vertices = num_vertices

        # build transformers
        self.transformers = nn.ModuleList([
            Transformer(hidden_dim=hidden_dims[i],
                        n_heads=n_heads,
                        depth=depth,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate)
            for i in range(num_layers)
        ])

        # self.pos_encodings = nn.ModuleList([
        #     build_position_encoding(pos_type='sine', hidden_dim=hidden_dims[i])
        #     for i in range(num_layers)
        # ])

        # Conv 1x1
        self.conv_1x1 = nn.ModuleList([
            nn.Conv2d(features_sizes[i], hidden_dims[i], kernel_size=1)
            for i in range(num_layers)
        ])

    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 X batch_size X 512
        joint_tokens = self.joint_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # num_j X batch_size X 512

        # extract image features through a CNN backbone
        img_features_list = self.backbone(images)  # batch_size X 2048 X h X w
        cam_features, enc_img_features, joint_features = None, None, None
        for i, img_features in enumerate(img_features_list):
            _, _, h, w = img_features.shape
            img_features = self.conv_1x1[i](img_features).flatten(2).permute(2, 0, 1)  # (h * w) X batch_size X f_size
            if i == 0:
                enc_img_features = img_features
            else:
                enc_img_features = torch.cat([enc_img_features, img_features], dim=0)

            # positional encodings
            pos_enc = self.pos_encodings[i](batch_size, h, w,
                                            device).flatten(2).permute(2, 0, 1)  # (h * w) X batch_size X hid_dim[i]

            # transformer encoder-decoder
            cam_features, enc_img_features, joint_features = self.transformers[i](enc_img_features, cam_token,
                                                                                  joint_tokens,
                                                                                  pos_enc, )
            # progressive dimensionality reduction
            cam_features = self.dim_reduce_enc_cam[i](cam_features)  # 1 X batch_size X hid_dim[i + 1]
            enc_img_features = self.dim_reduce_enc_img[i](enc_img_features)  # (h * w) X batch_size X hid_dim[i + 1]
            joint_features = self.dim_reduce_dec[i](joint_features)  # num_j X batch_size X hid_dim[i + 1]

        # estimators
        pred_cam = self.cam_predictor(cam_features).view(batch_size, 3)  # batch_size X 3
        pred_3d_joints = self.xyz_regressor(joint_features.transpose(0, 1))  # batch_size X num_joints X 3

        return {'pred_cam': pred_cam, 'pred_3d_joints': pred_3d_joints}
