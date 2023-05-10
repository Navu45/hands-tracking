import torch
from torch import nn

from core.ops import SeparableConv2d


class MultiscaleFusion(nn.Module):
    def __init__(self, in_channels_list, use_down_path):
        super().__init__()
        self.downsampling = nn.ModuleList([
            SeparableConv2d(in_channels_list[i - 1], in_channels_list[i - 1], 3, 2, 1,
                            norm_type='BN')
            for i in range(1, len(in_channels_list))
        ])
        self.use_down_path = use_down_path

    def forward(self, multiscale_x: list[torch.Tensor]):
        output = [multiscale_x[0]]
        for i in range(1, len(multiscale_x)):
            output.append(torch.cat([
                multiscale_x[i],
                self.downsampling[i - 1](multiscale_x[i - 1])
            ], dim=1))
        if not self.use_down_path:
            output.reverse()
        return output


class MultiScaleFusionTransformerLayer(nn.Module):  # Queen fusion
    def __init__(self, in_channels_list, out_channels_list, use_down_path=True):
        super().__init__()
        self.use_down_path = use_down_path

        if self.use_down_path:
            self.updown_path = nn.ModuleList([
                SeparableConv2d(out_channels_list[i], out_channels_list[i], 3, 2,
                                norm_type='BN')
                for i in range(1, len(out_channels_list))
            ])
        else:
            self.updown_path = nn.UpsamplingNearest2d(scale_factor=2)

        self.multiscale_fusion = MultiscaleFusion(in_channels_list, use_down_path)

        # Transformer block
        transformers = [
            nn.Identity()
            # (in_channels_list[i], out_channels_list[i])
            for i in range(len(in_channels_list))
        ]
        self.transformers = nn.ModuleList(transformers)

    def forward_updown_path(self, index, prev_fusion_block_out):
        if self.use_down_path:
            return self.updown_path[index](prev_fusion_block_out)
        return self.updown_path(prev_fusion_block_out)

    def forward(self, multiscale_x: list[torch.Tensor]):
        """
        Relative sizes of multiscale input can be described as this array: [(C, HW), (2C, HW / 2), (4C, HW / 4)]
        """
        multiscale_x = self.multiscale_fusion(multiscale_x)
        output = []
        for i, fusion_block in enumerate(self.transformers):
            x = [multiscale_x[i]]
            if len(output) != 0:
                x.append(self.forward_updown_path(i, output[-1]))
            x = torch.cat(x, dim=1)
            output.append(self.transformers[i](x))
        return output


if __name__ == '__main__':
    c = 64
    i = 512
    layer = MultiScaleFusionTransformerLayer(
        [4 * c, 8 * c, 16 * c],
        [4 * c, 8 * c, 16 * c],
        use_down_path=False
    )
    print(layer)
    layer_output = layer([
        torch.rand(10, 2 * c, i // 16, i // 16),
        torch.rand(10, 4 * c, i // 32, i // 32),
        torch.rand(10, 8 * c, i // 64, i // 64)
    ])
    print(*[t.shape for t in layer_output], sep='\n')

# class FastMETRO_JointsOnly(pl.LightningModule):
#     def __init__(self,
#                  features_sizes: list,
#                  hidden_dims: list,
#                  backbone,
#                  num_layers=3,
#                  num_joints=21,
#                  # num_vertices=195,
#                  n_heads=8,
#                  depth=2,
#                  mlp_ratio=2.0,
#                  drop_rate=0.1):
#         super().__init__()
#         self.backbone = backbone
#         self.num_joints = num_joints
#         # self.num_vertices = num_vertices
#
#         if len(hidden_dims) != num_layers or len(features_sizes) != num_layers:
#             raise ValueError("Specify connections between every transformer and backbone layer! num_layers is number "
#                              "of connections between transformer and backbone")
#
#         # token embeddings
#         self.cam_token_embed = nn.Embedding(1, hidden_dims[0])
#         self.joint_token_embed = nn.Embedding(num_joints, hidden_dims[0])
#         # self.vertex_token_embed = nn.Embedding(num_vertices, hidden_dims[0])
#
#         # build transformers
#         self.transformers = nn.ModuleList([
#             Transformer(hidden_dim=hidden_dims[i],
#                         n_heads=n_heads,
#                         depth=depth,
#                         mlp_ratio=mlp_ratio,
#                         drop_rate=drop_rate)
#             for i in range(num_layers)
#         ])
#
#         # self.pos_encodings = nn.ModuleList([
#         #     build_position_encoding(pos_type='sine', hidden_dim=hidden_dims[i])
#         #     for i in range(num_layers)
#         # ])
#
#         self.dim_reduce_enc_cam = nn.ModuleList([
#             nn.Linear(hidden_dims[i], hidden_dims[i + 1] if i < num_layers - 1 else hidden_dims[i])
#             for i in range(num_layers)
#         ])
#         self.dim_reduce_enc_img = nn.ModuleList([
#             nn.Linear(hidden_dims[i], hidden_dims[i + 1] if i < num_layers - 1 else hidden_dims[i])
#             for i in range(num_layers)
#         ])
#         self.dim_reduce_dec = nn.ModuleList([
#             nn.Linear(hidden_dims[i], hidden_dims[i + 1] if i < num_layers - 1 else hidden_dims[i])
#             for i in range(num_layers)
#         ])
#
#         self.xyz_regressor = nn.Linear(hidden_dims[-1], 3)
#         self.cam_predictor = nn.Linear(hidden_dims[-1], 3)
#
#         # Conv 1x1
#         self.conv_1x1 = nn.ModuleList([
#             nn.Conv2d(features_sizes[i], hidden_dims[i], kernel_size=1)
#             for i in range(num_layers)
#         ])
#
#     def forward(self, images):
#         device = images.device
#         batch_size = images.size(0)
#
#         # preparation
#         cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 X batch_size X 512
#         joint_tokens = self.joint_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # num_j X batch_size X 512
#
#         # extract image features through a CNN backbone
#         img_features_list = self.backbone(images)  # batch_size X 2048 X h X w
#         cam_features, enc_img_features, joint_features = None, None, None
#         for i, img_features in enumerate(img_features_list):
#             _, _, h, w = img_features.shape
#             img_features = self.conv_1x1[i](img_features).flatten(2).permute(2, 0, 1)  # (h * w) X batch_size X f_size
#             if i == 0:
#                 enc_img_features = img_features
#             else:
#                 enc_img_features = torch.cat([enc_img_features, img_features], dim=0)
#
#             # positional encodings
#             pos_enc = self.pos_encodings[i](batch_size, h, w,
#                                             device).flatten(2).permute(2, 0, 1)  # (h * w) X batch_size X hid_dim[i]
#
#             # transformer encoder-decoder
#             cam_features, enc_img_features, joint_features = self.transformers[i](enc_img_features, cam_token,
#                                                                                   joint_tokens,
#                                                                                   pos_enc, )
#             # progressive dimensionality reduction
#             cam_features = self.dim_reduce_enc_cam[i](cam_features)  # 1 X batch_size X hid_dim[i + 1]
#             enc_img_features = self.dim_reduce_enc_img[i](enc_img_features)  # (h * w) X batch_size X hid_dim[i + 1]
#             joint_features = self.dim_reduce_dec[i](joint_features)  # num_j X batch_size X hid_dim[i + 1]
#
#         # estimators
#         pred_cam = self.cam_predictor(cam_features).view(batch_size, 3)  # batch_size X 3
#         pred_3d_joints = self.xyz_regressor(joint_features.transpose(0, 1))  # batch_size X num_joints X 3
#
#         return {'pred_cam': pred_cam, 'pred_3d_joints': pred_3d_joints}
