import torch
from torch import nn
from policy.minkowski.resnet import ResNet14


class Sparse3DEncoder(torch.nn.Module):
    def __init__(self, input_dim = 6, output_dim = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cloud_encoder = ResNet14(in_channels=input_dim, out_channels=output_dim, conv1_kernel_size=3, dilations=(1,1,1,1), bn_momentum=0.02)
        self.position_embedding = SparsePositionalEncoding(output_dim)

    def forward(self, sinput, max_num_token=100, batch_size=24):
        ''' max_num_token: maximum token number for each point cloud, which can be adjusted depending on the scene density.
                           100 for voxel_size=0.005 in our experiments
        '''
        soutput = self.cloud_encoder(sinput)
        feats_batch, coords_batch = soutput.F, soutput.C
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:,0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype=torch.bool, device=feats_batch.device)
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            num_token = min(max_num_token, len(feats))
            tokens[i,:num_token] = feats[:num_token]
            pos_emb[i,:num_token] = pos[:num_token]
            token_padding_mask[i,:num_token] = False
        
        return tokens, pos_emb, token_padding_mask

# class Dense3DEncoder(nn.Module):
#     def __init__(self, input_dim=6, output_dim=512, max_num_token=100, batch_size=24, max_voxel_dim=64):
#         super().__init__()
#         self.output_dim = output_dim
#         self.max_num_token = max_num_token
#         self.batch_size = batch_size

#         self.encoder = nn.Sequential(
#             nn.Conv3d(input_dim, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#             nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU(),
#             nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#             nn.Conv3d(256, output_dim, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(output_dim),
#             nn.ReLU()
#         )
#         self.position_embedding = DensePositionalEncoding(num_pos_feats=output_dim, max_pos=max_voxel_dim)

#     def forward(self, x):
#         """
#         x: (B, C_in, D, H, W) dense 3D voxel tensor
#         Returns:
#             tokens: (B, max_num_token, output_dim)
#             pos_emb: (B, max_num_token, output_dim)
#             token_padding_mask: (B, max_num_token)
#         """
#         B, C, D, H, W = x.shape
#         features = self.encoder(x)  # (B, output_dim, D', H', W')
#         _, _, Dp, Hp, Wp = features.shape

#     # Flatten features
#         features = features.flatten(2).transpose(1, 2)  # (B, N, D)
#         N = features.shape[1]

#     # Positional encoding
#         pos = self.position_embedding(Dp, Hp, Wp, x.device)
#         pos = pos.flatten(1).permute(1, 0).unsqueeze(0).repeat(B, 1, 1)  # (B, N, D)

#     # Truncate or pad to max_num_token
#         num_token = min(self.max_num_token, N)
#         tokens = torch.zeros(B, self.max_num_token, self.output_dim, device=x.device)
#         pos_emb = torch.zeros_like(tokens)
#         mask = torch.ones(B, self.max_num_token, dtype=torch.bool, device=x.device)

#         tokens[:, :num_token] = features[:, :num_token]
#         pos_emb[:, :num_token] = pos[:, :num_token]
#         mask[:, :num_token] = False

#         return tokens, pos_emb, mask

class SparsePositionalEncoding(nn.Module):
    """
    Sparse positional encoding for point tokens, similar to the standard version
    """
    def __init__(self, num_pos_feats=64, temperature=10000, max_pos=800):
        super().__init__()
        ''' max_pos: position range will be [-max_pos/2, max_pos/2) along X/Y/Z-axis.
                     remeber to keep this value fixed in your training and evaluation.
                     800 for voxel_size=0.005 in our experiments.
        '''
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]

        x_dim_t = torch.arange(x_steps, dtype=torch.float32)
        y_dim_t = torch.arange(y_steps, dtype=torch.float32)
        z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps)
        y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps)
        z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)

        pos_x_vector = xyz_embed / x_dim_t
        pos_y_vector = xyz_embed / y_dim_t
        pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1)
            coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1)
            coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]
            pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]
            pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list
    
# class DensePositionalEncoding(nn.Module):
#     def __init__(self, num_pos_feats=64, temperature=10000, max_pos=100):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.max_pos = max_pos
#         self.origin_pos = max_pos // 2
#         self._init_position_vector()

#     def _init_position_vector(self):
#         x_steps = y_steps = self.num_pos_feats // 3
#         z_steps = self.num_pos_feats - x_steps - y_steps
#         xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:, None]

#         x_dim_t = self.temperature ** (2 * (torch.arange(x_steps) // 2) / x_steps)
#         y_dim_t = self.temperature ** (2 * (torch.arange(y_steps) // 2) / y_steps)
#         z_dim_t = self.temperature ** (2 * (torch.arange(z_steps) // 2) / z_steps)

#         pos_x_vector = xyz_embed / x_dim_t
#         pos_y_vector = xyz_embed / y_dim_t
#         pos_z_vector = xyz_embed / z_dim_t

#         self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
#         self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
#         self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

#     def forward(self, D, H, W, device):
#         z_range = torch.arange(D, device=device) + self.origin_pos
#         y_range = torch.arange(H, device=device) + self.origin_pos
#         x_range = torch.arange(W, device=device) + self.origin_pos

#         z_range = torch.clamp(z_range, 0, self.max_pos - 1)
#         y_range = torch.clamp(y_range, 0, self.max_pos - 1)
#         x_range = torch.clamp(x_range, 0, self.max_pos - 1)

#         pos_z = self.pos_z_vector.to(device)[z_range]
#         pos_y = self.pos_y_vector.to(device)[y_range]
#         pos_x = self.pos_x_vector.to(device)[x_range]

#         pos = pos_z[:, None, None, :] + pos_y[None, :, None, :] + pos_x[None, None, :, :]
#         return pos.permute(3, 0, 1, 2)