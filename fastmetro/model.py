import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from fastmetro import build_position_encoding

class HandJointReconstructor(pl.LightningModule):
    def __init__(self,
                 features_size,
                 hidden_dim,
                 backbone,
                 num_joints=21,
                 num_layers=3,
                 ):
        super().__init__()
        self.backbone = backbone

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, hidden_dim)
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])

        self.pos_encod = build_position_encoding(pos_type='sine', hidden_dim=hidden_dim)

        self.xyz_regressor = nn.Linear(hidden_dim, 3)
        self.cam_predictor = nn.Linear(hidden_dim, 3)

        # Conv 1x1
        self.conv_1x1 = nn.Conv2d(features_size, hidden_dim, kernel_size=1)

        # # attention mask
        # zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool))
        # zeros_2 = torch.tensor(np.zeros((num_joints, num_joints)).astype(bool))
        # adjacency_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
        # adjacency_matrix_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
        # adjacency_matrix_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')
        # adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value,
        #                                            size=adjacency_matrix_size).to_dense()
        # temp_mask_1 = (adjacency_matrix == 0)
        # temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        # self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

