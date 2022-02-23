import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import Mlp_plus_t_emb


class Pnet2Stage(nn.Module):
    def __init__(self, mlp1, mlp2, bn=True, remove_last_activation=True):
        super().__init__()
        self.mlp1 = Mlp_plus_t_emb(mlp1, bn=bn, t_dim=0, include_t=False,
                        bn_first=False, bias=True, first_conv=False,
                        first_conv_in_channel=0,
                        res_connect=False,
                        include_condition=False, condition_dim=128)
        if remove_last_activation:
            self.mlp1.second_mlp = self.mlp1.second_mlp[0:1]

        mlp2 = [2*mlp1[-1]] +  mlp2
        self.mlp2 = Mlp_plus_t_emb(mlp2, bn=bn, t_dim=0, include_t=False,
                        bn_first=False, bias=True, first_conv=False,
                        first_conv_in_channel=0,
                        res_connect=False,
                        include_condition=False, condition_dim=128)
        if remove_last_activation:
            self.mlp2.second_mlp = self.mlp2.second_mlp[0:1]

    def forward(self, x):
        # x should be of size (B, mlp1[0], num_points)
        x_feature = x.unsqueeze(-1) # shape (B, mlp1[0], num_points, 1)
        feature = self.mlp1(x_feature)
        # feature is of shape (B, mlp1[-1], num_points, 1)
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])
        # global_feature is of shape (B, mlp1[-1], 1, 1)
        global_feature = global_feature.expand(-1,-1,feature.size(2),-1)
        feature = torch.cat([feature, global_feature], dim=1)

        feature = self.mlp2(feature)
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])
        global_feature = global_feature.squeeze(-1).squeeze(-1)
        return global_feature

import pdb
if __name__ == '__main__':
    
    mlp1 = [3, 128, 256]
    mlp2 = [512, 1024]
    pnet = Pnet2Stage(mlp1, mlp2)
    print(pnet)
    x = torch.rand(16, 3, 2048)*2-1
    # x = x.unsqueeze(-1)
    feature = pnet(x)
    pdb.set_trace()