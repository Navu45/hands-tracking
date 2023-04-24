import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from fastmetro import build_position_encoding
from fastmetro.transformer import Transformer


class HandReconstructor(pl.LightningModule):
    def __init__(self,
                 features_sizes: list,
                 hidden_dims: list,
                 backbone,
                 num_layers=2,
                 num_joints=21,
                 num_vertices=195,
                 n_heads=8,
                 depth=2,
                 mlp_ratio=2.0,
                 drop_rate=0.1):
        super().__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        if len(hidden_dims) != num_layers or len(features_sizes) != num_layers:
            raise ValueError("Specify connections between every transformer and backbone layer! num_layers is number "
                             "of connections between transformer and backbone")

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, hidden_dims[0])
        self.joint_token_embed = nn.Embedding(num_joints, hidden_dims[0])
        self.vertex_token_embed = nn.Embedding(num_vertices, hidden_dims[0])

        # build transformers
        self.transformers = nn.ModuleList([
            Transformer(hidden_dim=hidden_dims[i],
                        n_heads=n_heads,
                        depth=depth,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate)
            for i in range(num_layers)
        ])

        self.pos_encodings = nn.ModuleList([
            build_position_encoding(pos_type='sine', hidden_dim=hidden_dims[i])
            for i in range(num_layers)
        ])

        self.dim_reduce_enc_cam = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1] if num_layers > 1 else hidden_dims[i])
            for i in range(0, num_layers)
        ])
        self.dim_reduce_enc_img = nn.ModuleList([
            nn.Linear(hidden_dims[i - 1], hidden_dims[i] if num_layers > 1 else hidden_dims[i - 1])
            for i in range(0, num_layers)
        ])
        self.dim_reduce_dec = nn.ModuleList([
            nn.Linear(hidden_dims[i - 1], hidden_dims[i] if num_layers > 1 else hidden_dims[i - 1])
            for i in range(0, num_layers)
        ])

        self.xyz_regressor = nn.Linear(hidden_dims[0], 3)
        self.cam_predictor = nn.Linear(hidden_dims[0], 3)

        # Conv 1x1
        self.conv_1x1 = nn.ModuleList([
            nn.Conv2d(features_sizes[i], hidden_dims[i], kernel_size=1)
            for i in range(num_layers)
        ])

        # attention mask
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool))
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool))
        adjacency_indices = torch.load('data/mano/mano_195_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('data/mano/mano_195_adjmat_values.pt')
        adjacency_matrix_size = torch.load('data/mano/mano_195_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value,
                                                   size=adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 X batch_size X 512
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(
            1).repeat(1, batch_size, 1)  # (num_j + num_v) X batch_size X 512
        attention_mask = self.attention_mask.to(device)  # (num_j + num_v) X (num_j + num_v)

        # extract image features through a CNN backbone
        img_features_list = self.backbone(images)  # batch_size X 2048 X 7 X 7
        cam_features, enc_img_features, jv_features = None, None, None
        for i, img_features in enumerate(img_features_list):
            _, _, h, w = img_features.shape
            img_features = self.conv_1x1[i](img_features).flatten(2).permute(2, 0, 1)  # 49 X batch_size X ftrs_size
            if i == 0:
                enc_img_features = img_features
            else:
                enc_img_features = torch.cat([enc_img_features, img_features], dim=0)

            # positional encodings
            pos_enc = self.pos_encodings[i](batch_size, h, w,
                                            device).flatten(2).permute(2, 0, 1)  # 49 X batch_size X hid_dim[i]

            # transformer encoder-decoder
            cam_features, enc_img_features, jv_features = self.transformers[i](enc_img_features, cam_token, jv_tokens,
                                                                               pos_enc,
                                                                               attention_mask=attention_mask)
            # progressive dimensionality reduction
            cam_features = self.dim_reduce_enc_cam[i](cam_features)  # 1 X batch_size X hid_dim[i + 1]
            enc_img_features = self.dim_reduce_enc_img[i](enc_img_features)  # 49 X batch_size X hid_dim[i + 1]
            jv_features = self.dim_reduce_dec[i](jv_features)  # (num_j + num_v) X batch_size X hid_dim[i + 1]

        # estimators
        pred_cam = self.cam_predictor(cam_features).view(batch_size, 3)  # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(
            jv_features.transpose(0, 1))  # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, :self.num_joints, :]  # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.num_joints:, :]  # batch_size X num_vertices(coarse) X 3

        return {'pred_cam': pred_cam,
                'pred_3d_joints': pred_3d_joints,
                'pred_3d_vertices_coarse': pred_3d_vertices_coarse}
