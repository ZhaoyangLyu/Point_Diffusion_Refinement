from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
# import pointnet2_utils
# from pointnet2_ops.attention import AttentionModule
from pointnet2_ops.attention import AttentionModule, GlobalAttentionModule

import copy

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out

def build_shared_mlp(mlp_spec: List[int], bn: bool = True, 
                        bn_first: bool = False,
                        bias: bool = False,
                        activation: str = 'relu'):
    assert activation in ['relu', 'swish']
    layers = []
    for i in range(1, len(mlp_spec)):
        if bn_first:
            if bn:
                # layers.append(nn.BatchNorm2d(mlp_spec[i]))
                layers.append(MyGroupNorm(min(32, mlp_spec[i-1]), mlp_spec[i-1]))
            if activation == 'relu':
                layers.append(nn.ReLU(True))
            elif activation == 'swish':
                layers.append(Swish())
        layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=bias))
        if not bn_first:
            if bn:
                # layers.append(nn.BatchNorm2d(mlp_spec[i]))
                layers.append(MyGroupNorm(32, mlp_spec[i]))
            if activation == 'relu':
                layers.append(nn.ReLU(True))
            elif activation == 'swish':
                layers.append(Swish())

    return nn.Sequential(*layers)

class Mlp_plus_t_emb(nn.Module):
    def __init__(self, mlp_spec, bn, t_dim=128, include_t=True,
                        bn_first=False, bias=False, first_conv=False,
                        first_conv_in_channel=0,
                        res_connect=False,
                        include_condition=False, condition_dim=128,
                        include_second_condition=False, second_condition_dim=128,
                        activation='relu'):
        super(Mlp_plus_t_emb, self).__init__()

        self.include_t = include_t
        if include_t:
            self.fc = nn.Linear(t_dim, mlp_spec[1])

        self.include_condition = include_condition
        if include_condition:
            self.fc_condition = nn.Linear(condition_dim, mlp_spec[2])

        self.include_second_condition = include_second_condition
        if include_second_condition:
            self.fc_second_condition = nn.Linear(second_condition_dim, mlp_spec[-1])

        self.first_conv_bool = first_conv
        if first_conv:
            self.first_conv = nn.Conv2d(first_conv_in_channel, 
                        mlp_spec[0], kernel_size=1, bias=bias)

        self.res_connect_bool = res_connect
        if res_connect:
            if mlp_spec[0] == mlp_spec[-1]:
                self.res_connect = None
            else:
                self.res_connect = nn.Conv2d(mlp_spec[0], mlp_spec[-1], kernel_size=1, bias=bias)

        # mlp_spec should be at least of length 3
        assert len(mlp_spec) >= 3
        if include_second_condition:
            assert len(mlp_spec) >= 4
        self.first_mlp = build_shared_mlp(mlp_spec[0:2], bn,
                        bn_first=bn_first, bias=bias, activation=activation)
        self.second_mlp = build_shared_mlp(mlp_spec[1:3], bn,
                        bn_first=bn_first, bias=bias, activation=activation)
        if len(mlp_spec) > 3:
            self.rest_mlp = build_shared_mlp(mlp_spec[2:], bn,
                        bn_first=bn_first, bias=bias, activation=activation)
        else:
            self.rest_mlp = None

    def forward(self, feature, t_emb=None, condition_emb=None, second_condition_emb=None):
        # feature is of shape (B, fea_in, npoint, nsample)
        # t_emb is of shape (B, t_dim)
        
        if self.first_conv_bool:
            feature = self.first_conv(feature)
            # print('first conv used')
            # feature is of shape (B, mlp_spec[0], npoint, nsample)

        h = feature
        h = self.first_mlp(h)
        # h is of shape (B, mlp_spec[1], npoint, nsample)
        if self.include_t:
            if t_emb is None:
                raise Exception('Should pass t_emb to the forward function')
            else:
                t1 = self.fc(t_emb) # shape (B, mlp_spec[1])
                t1 = t1.unsqueeze(2).unsqueeze(3) # shape (B, mlp_spec[1], 1, 1)
                h = h + t1
                # print('t_emb used')
        else:
            if not t_emb is None:
                raise Exception('This module does not include t but t_emb is given')

        h = self.second_mlp(h)
        if self.include_condition:
            if condition_emb is None:
                raise Exception('Should pass condition_emb to the forward function')
            else:
                condition1 = self.fc_condition(condition_emb) # shape (B, mlp_spec[2])
                condition1 = condition1.unsqueeze(2).unsqueeze(3) # shape (B, mlp_spec[2], 1, 1)
                h = h + condition1
                # print('condition_emb used')
        else:
            if not condition_emb is None:
                raise Exception('This module does not include condition but condition_emb is given')


        if not self.rest_mlp is None:
            h = self.rest_mlp(h)
        if self.include_second_condition:
            if second_condition_emb is None:
                raise Exception('Should pass second_condition_emb to the forward function')
            else:
                condition2 = self.fc_second_condition(second_condition_emb) # shape (B, mlp_spec[2])
                condition2 = condition2.unsqueeze(2).unsqueeze(3) # shape (B, mlp_spec[2], 1, 1)
                h = h + condition2
        else:
            if not second_condition_emb is None:
                raise Exception('This module does not include condition but condition_emb is given')

        if self.res_connect_bool:
            if not self.res_connect is None:
                h = h + self.res_connect(feature)
                # print('res connect used')
            else:
                h = h + feature
        return h


def pooling_features(feature, count=None, pooling='max'):
    # feature is of shape (B, C, npoints, K)
    # we pool the feature into shape (B, C, npoints)
    assert pooling in ['max', 'avg', 'avg_max', 'max_avg']
    # avg_max and max_avg are the same
    if pooling == 'max':
        max_feature = F.max_pool2d(feature, kernel_size=[1, feature.size(3)])  
        # (B, mlp[-1], npoint, 1)
        max_feature = max_feature.squeeze(-1)  # (B, mlp[-1], npoint)
        return max_feature
    elif pooling == 'avg':
        K = feature.size(3)
        avg_feature = pointnet2_utils.average_feature(feature, count, K)
        return avg_feature
    elif 'avg' in pooling and 'max' in pooling:
        K = feature.size(3)
        C = feature.shape[1]
        half_C = int(C/2)
        feature1 = feature[:,0:half_C,:,:]
        feature2 = feature[:,half_C:,:,:]

        max_feature = F.max_pool2d(feature1, kernel_size=[1, K])  
        # (B, mlp[-1], npoint, 1)
        max_feature = max_feature.squeeze(-1)  # (B, mlp[-1], npoint)

        avg_feature = pointnet2_utils.average_feature(feature2, count, K)
        new_feature = torch.cat([max_feature, avg_feature], dim=1)
        return new_feature
    else:
        raise Exception("%s pooling is not supported" % pooling)
        


class _PointnetSAModuleBase(nn.Module):
    # set abstraction module, down sampling
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.inlcude_t = None
        self.t_dim = None

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor], 
                t_emb: torch.Tensor=None, condition_emb: torch.Tensor=None, second_condition_emb: torch.Tensor=None, 
                subset: bool=True, record_neighbor_stats: bool=False, pooling: str='max') -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        t_emb : torch.Tensor
            (B, t_dim) tensor of the time step

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        # reduce the number of points per shape from N to npoint
        # each new point in the npoint points have nsample neighbors in the original N points

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous() # shape (B,3,N)
        
        assert self.npoint is not None
        furthest_point_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, furthest_point_idx).transpose(1, 2).contiguous()
        # shape (B, npoint, 3)
        if self.use_attention_module:
            new_xyz_feat = pointnet2_utils.gather_operation(features, furthest_point_idx)
            # shape (B, C, npoint), features at new_xyz

        for i in range(len(self.groupers)):
            grouped_features, count = self.groupers[i](xyz, new_xyz, features, subset=subset, 
                                record_neighbor_stats=record_neighbor_stats, return_counts=True)  # (B, C+3, npoint, nsample)
            
            t_emb = t_emb if self.include_t else None
            condition_emb = condition_emb if self.include_condition else None
            second_condition_emb = second_condition_emb if self.include_second_condition else None
            out_features = self.mlps[i](grouped_features, t_emb=t_emb, condition_emb=condition_emb,
                                                second_condition_emb=second_condition_emb)
            # (B, mlp[-1], npoint, K)
            
            if self.use_attention_module:
                new_features = self.attention_modules[i](new_xyz_feat, grouped_features, out_features, count)
                # (B, mlp[-1], npoint)
            else:
                new_features = pooling_features(out_features, count=count, pooling=pooling)
                # (B, mlp[-1], npoint)
            if self.use_global_attention_module:
                new_xyz_flipped = new_xyz.transpose(1, 2) # shape (B,3,npoint)
                new_features = torch.cat([new_features, new_xyz_flipped], dim=1) # (B, mlp[-1]+3, npoint)
                new_features = self.global_attention_modules[i](new_features) # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True, 
                        t_dim=128, include_t=False, include_abs_coordinate=False, include_center_coordinate=False,
                        bn_first=False, bias=False, first_conv=False,
                        first_conv_in_channel=0, res_connect=False,
                        include_condition=False, condition_dim=128,
                        include_second_condition=False, second_condition_dim=128,
                        neighbor_def='radius', activation='relu', attention_setting=None,
                        global_attention_setting=None):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        # use_xyz whether use xyz as feature during forward
        # if use_xyz is true, we will concat xyz with the given feature during the forward pass
        
        super(PointnetSAModuleMSG, self).__init__()

        self.include_t = include_t
        self.t_dim = t_dim
        self.include_condition = include_condition
        self.condition_dim = condition_dim
        self.include_second_condition = include_second_condition
        self.second_condition_dim = second_condition_dim
        # self.include_abs_coordinate = include_abs_coordinate

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        self.use_attention_module = False
        if attention_setting is not None:
            assert isinstance(attention_setting, dict)
            self.use_attention_module = attention_setting['use_attention_module']
        self.attention_modules = nn.ModuleList() if self.use_attention_module else None

        self.use_global_attention_module = False
        if global_attention_setting is not None:
            assert isinstance(global_attention_setting, dict)
            self.use_global_attention_module = global_attention_setting['use_global_attention_module']
        self.global_attention_modules = nn.ModuleList() if self.use_global_attention_module else None

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, 
                                            include_abs_coordinate=include_abs_coordinate,
                                            include_center_coordinate=include_center_coordinate,
                                            neighbor_def=neighbor_def)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]

            ori_first_conv_in_channel = copy.deepcopy(first_conv_in_channel)
            ori_mlp_spec0 = copy.deepcopy(mlp_spec[0])

            if first_conv:
                if use_xyz:
                    first_conv_in_channel += 3
                    if include_abs_coordinate:
                        first_conv_in_channel += 3
                    if include_center_coordinate:
                        first_conv_in_channel += 3
            else:
                if use_xyz:
                    mlp_spec[0] += 3
                    if include_abs_coordinate:
                        mlp_spec[0] += 3
                    if include_center_coordinate:
                        mlp_spec[0] += 3

            
            self.mlps.append(Mlp_plus_t_emb(mlp_spec, bn, t_dim=self.t_dim, include_t=include_t,
                        bn_first=bn_first, bias=bias, first_conv=first_conv,
                        first_conv_in_channel=first_conv_in_channel, res_connect=res_connect,
                        include_condition=include_condition, condition_dim=condition_dim,
                        include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                        activation=activation))
            
            if self.use_attention_module:
                C_in1 = ori_first_conv_in_channel if first_conv else ori_mlp_spec0
                C_in2 = first_conv_in_channel if first_conv else mlp_spec[0]
                C1 = C_in1; C2 = C_in2
                C_out = mlp_spec[-1]
                self.attention_modules.append(AttentionModule(C_in1, C_in2, C1, C2, C_out, 
                    attention_bn=attention_setting['attention_bn'], 
                    transform_grouped_feat_out=attention_setting['transform_grouped_feat_out'], 
                    last_activation=attention_setting['last_activation']))
            
            if self.use_global_attention_module:
                self.global_attention_modules.append(GlobalAttentionModule(mlp_spec[-1], 
                additional_dim=3, attention_bn=global_attention_setting['attention_bn'],
                last_activation=global_attention_setting['last_activation']))



class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True,
                    t_dim=128, include_t=False, include_abs_coordinate=False,
                    include_center_coordinate=False,
                    bn_first=False, bias=False, first_conv=False,
                    first_conv_in_channel=0, res_connect=False,
                    include_condition=False, condition_dim=128, 
                    include_second_condition=False, second_condition_dim=128,
                    neighbor_def='radius', activation='relu',
                    attention_setting=None,
                    global_attention_setting=None):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            t_dim=t_dim,
            include_t = include_t,
            include_abs_coordinate=include_abs_coordinate,
            include_center_coordinate=include_center_coordinate,
            bn_first=bn_first, bias=bias, first_conv=first_conv,
            first_conv_in_channel=first_conv_in_channel, res_connect=res_connect,
            include_condition=include_condition, condition_dim=condition_dim,
            include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
            neighbor_def=neighbor_def, activation=activation,
            attention_setting=attention_setting,
            global_attention_setting=global_attention_setting
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    PointnetFPModule has no attention mechanism because it adopts three interpolate
    """

    def __init__(self, mlp, bn=True, t_dim=128, include_t=False,
                    bn_first=False, bias=False, first_conv=False,
                    first_conv_in_channel=0, res_connect=False,
                    include_condition=False, condition_dim=128,
                    include_second_condition=False, second_condition_dim=128,
                    include_grouper = False,
                    radius=0, nsample=32, use_xyz=True, 
                    include_abs_coordinate=True,
                    include_center_coordinate = False,
                    neighbor_def='radius', activation='relu'):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.include_t = include_t
        self.t_dim = t_dim
        self.include_condition = include_condition
        self.condition_dim = condition_dim
        self.include_second_condition = include_second_condition
        self.second_condition_dim = second_condition_dim
        
        
        self.include_grouper = include_grouper
        if self.include_grouper:
            # self.include_abs_coordinate = include_abs_coordinate
            # self.use_xyz = use_xyz
            if first_conv:
                if use_xyz:
                    first_conv_in_channel += 3
                    if include_abs_coordinate:
                        first_conv_in_channel += 3
                    if include_center_coordinate:
                        first_conv_in_channel += 3
            else:
                if use_xyz:
                    mlp[0] += 3
                    if include_abs_coordinate:
                        mlp[0] += 3
                    if include_center_coordinate:
                        mlp[0] += 3
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, 
                                            include_abs_coordinate=include_abs_coordinate,
                                            include_center_coordinate=include_center_coordinate,
                                            neighbor_def=neighbor_def)

        self.mlp = Mlp_plus_t_emb(mlp, bn, t_dim=self.t_dim, include_t=include_t,
                        bn_first=bn_first, bias=bias, first_conv=first_conv,
                        first_conv_in_channel=first_conv_in_channel, res_connect=res_connect,
                        include_condition=include_condition, condition_dim=condition_dim,
                        include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                        activation=activation)

    def forward(self, unknown, known, unknow_feats, known_feats, t_emb=None, condition_emb=None, 
                    second_condition_emb = None,
                    record_neighbor_stats=False, pooling='max'):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        # known_feats are features at positions specified at known
        # unknow_feats are features at positions specified at unknown
        # unknow_feats features at the positons unknown obtained in the encoder

        # we first interpolate known_feats to interpolated_feats at positions unknown
        # then concat interpolated_feats with unknow_feats to get new_features
        # then new_features is put through a pointnet 
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        # pdb.set_trace()
        if self.include_grouper:
            new_features, count = self.grouper(unknown, unknown, new_features, subset=True, 
                                record_neighbor_stats=record_neighbor_stats, return_counts=True)
        else:
            new_features = new_features.unsqueeze(-1)
        
        t_emb = t_emb if self.include_t else None
        condition_emb = condition_emb if self.include_condition else None
        second_condition_emb = second_condition_emb if self.include_second_condition else None
        new_features = self.mlp(new_features, t_emb=t_emb, condition_emb=condition_emb,
                                            second_condition_emb=second_condition_emb)

        if self.include_grouper:
            new_features = pooling_features(new_features, count=count, pooling=pooling)
            return new_features
            # new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  
            # (B, mlp[-1], npoint, 1)
        
        return new_features.squeeze(-1)


class FeatureMapModule(nn.Module):
    def __init__(self, mlp, radius, K, use_xyz=True, include_abs_coordinate=True, include_center_coordinate=False,
                    bn=True, bn_first=True, bias=True, res_connect=True,
                    first_conv=False, first_conv_in_channel=0, neighbor_def = 'radius', activation='relu', attention_setting=None,
                    query_feature_dim=None):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(FeatureMapModule, self).__init__()

        self.use_attention_module = False
        if attention_setting is not None:
            assert isinstance(attention_setting, dict)
            self.use_attention_module = attention_setting['use_attention_module']
        # self.attention_modules = nn.ModuleList() if self.use_attention_module else None
        # ori_first_conv_in_channel = copy.deepcopy(first_conv_in_channel)
        # ori_mlp_spec0 = copy.deepcopy(mlp_spec[0])

        if first_conv:
            if use_xyz:
                first_conv_in_channel += 3
                if include_abs_coordinate:
                    first_conv_in_channel += 3
                if include_center_coordinate:
                    first_conv_in_channel += 3
        else:
            if use_xyz:
                mlp[0] += 3
                if include_abs_coordinate:
                    mlp[0] += 3
                if include_center_coordinate:
                    mlp[0] += 3

        self.mlp = Mlp_plus_t_emb(mlp, bn, include_t=False,
                        bn_first=bn_first, bias=bias, first_conv=first_conv,
                        first_conv_in_channel=first_conv_in_channel, res_connect=res_connect,
                        include_condition=False, activation=activation)
        self.mapper = pointnet2_utils.QueryAndGroup(radius, K, use_xyz=use_xyz, 
                            include_abs_coordinate=include_abs_coordinate, 
                            include_center_coordinate=include_center_coordinate,
                            neighbor_def = neighbor_def)
        
        if self.use_attention_module:
            # C_in1 = ori_first_conv_in_channel if first_conv else ori_mlp_spec0
            C_in1 = query_feature_dim
            C_in2 = first_conv_in_channel if first_conv else mlp[0]
            C1 = C_in1; C2 = C_in2
            C_out = mlp[-1]
            self.attention_module = AttentionModule(C_in1, C_in2, C1, C2, C_out, 
                attention_bn=attention_setting['attention_bn'], 
                transform_grouped_feat_out=attention_setting['transform_grouped_feat_out'], 
                last_activation=attention_setting['last_activation'])

    def forward(self, xyz, features, new_xyz, subset=False, record_neighbor_stats=True, pooling='max',
                    features_at_new_xyz=None):
        # features are at position xyz
        # this function maps features to max_features at new_xyz
        # xyz (B,N,3), features (B,C,N)
        # new_xyz (B, npoint, 3), max_features (B,mlp[-1],npoint)
        # you should specify features_at_new_xyz if you want to use attention module
        # features_at_new_xyz is of shape B, C', npoint, it is features at point new_xyz, and is used as query for attention
        new_features, count = self.mapper(xyz, new_xyz, features, subset=subset, record_neighbor_stats=record_neighbor_stats,
                                    return_counts=True) 
        # (B, C, npoint, nsample)
        out_features = self.mlp(new_features) # (B, mlp[-1], npoint, nsample)

        if self.use_attention_module:
            max_features = self.attention_module(features_at_new_xyz, new_features, out_features, count)
        else:
            max_features = pooling_features(out_features, count=count, pooling=pooling)
        # max_features = F.max_pool2d(out_features, kernel_size=[1, out_features.size(3)]) # (B, mlp[-1], npoint, 1)
        # max_features = max_features.squeeze(-1) # (B, mlp[-1], npoint)
        return max_features


class PointnetKnnFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp1, mlp2, K, bn=True, t_dim=128, include_t=False,
                    bn_first=False, bias=False, first_conv=False,
                    first_conv_in_channel1=0, first_conv_in_channel2=0, res_connect=False,
                    include_condition=False, condition_dim=128,
                    include_second_condition=False, second_condition_dim=128,
                    include_grouper = False,
                    radius=0, nsample=32, use_xyz=True, 
                    include_abs_coordinate=True,
                    include_center_coordinate=False,
                    neighbor_def='radius', activation='relu',
                    attention_setting=None,
                    global_attention_setting=None):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetKnnFPModule, self).__init__()
        self.include_t = include_t
        self.t_dim = t_dim
        self.include_condition = include_condition
        self.condition_dim = condition_dim
        self.include_second_condition = include_second_condition
        self.second_condition_dim = second_condition_dim
        self.K = K # number of nearest neighbors
        
        if first_conv:
            first_conv_in_channel1 = first_conv_in_channel1 + 11
        else:
            mlp1[0] = mlp1[0] + 11
        self.mlp1 = Mlp_plus_t_emb(mlp1, bn, t_dim=self.t_dim, include_t=False,
                        bn_first=bn_first, bias=bias, first_conv=first_conv,
                        first_conv_in_channel=first_conv_in_channel1, res_connect=res_connect,
                        include_condition=include_second_condition, condition_dim=second_condition_dim,
                        activation=activation)
                        # include_second_condition=include_second_condition, second_condition_dim=second_condition_dim)
        
        # self attention mechanism is applied after the first mlp
        self.use_attention_module = False
        if attention_setting is not None:
            assert isinstance(attention_setting, dict)
            self.use_attention_module = attention_setting['use_attention_module']
        if self.use_attention_module:
            # C_in1 = ori_first_conv_in_channel1 if first_conv else ori_mlp1_spec0
            # unknown_feats are used as query
            C_in1 = first_conv_in_channel2-mlp1[-1] if first_conv else mlp2[0] - mlp1[-1] # the dimension of unknown_feats
            C_in2 = first_conv_in_channel1 if first_conv else mlp1[0]
            C1 = C_in1; C2 = C_in2
            C_out = mlp1[-1]
            self.attention_module = AttentionModule(C_in1, C_in2, C1, C2, C_out, 
                attention_bn=attention_setting['attention_bn'], 
                transform_grouped_feat_out=attention_setting['transform_grouped_feat_out'], 
                last_activation=attention_setting['last_activation'])

        self.include_grouper = include_grouper
        if self.include_grouper:
            # self.include_abs_coordinate = include_abs_coordinate
            # self.use_xyz = use_xyz
            if first_conv:
                if use_xyz:
                    first_conv_in_channel2 += 3
                    if include_abs_coordinate:
                        first_conv_in_channel2 += 3
                    if include_center_coordinate:
                        first_conv_in_channel2 += 3
            else:
                if use_xyz:
                    mlp2[0] += 3
                    if include_abs_coordinate:
                        mlp2[0] += 3
                    if include_center_coordinate:
                        mlp2[0] += 3

            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, 
                                            include_abs_coordinate=include_abs_coordinate,
                                            include_center_coordinate=include_center_coordinate,
                                            neighbor_def=neighbor_def)
        else:
            if first_conv:
                first_conv_in_channel2 = first_conv_in_channel2 + 3
            else:
                mlp2[0] = mlp2[0] + 3

        self.mlp2 = Mlp_plus_t_emb(mlp2, bn, t_dim=self.t_dim, include_t=include_t,
                        bn_first=bn_first, bias=bias, first_conv=first_conv,
                        first_conv_in_channel=first_conv_in_channel2, res_connect=res_connect,
                        include_condition=include_condition, condition_dim=condition_dim, activation=activation)
        
        # global attention mechanism is applied after the second mlp
        self.use_global_attention_module = False
        if global_attention_setting is not None:
            assert isinstance(global_attention_setting, dict)
            self.use_global_attention_module = global_attention_setting['use_global_attention_module']
        if self.use_global_attention_module:
            self.global_attention_module = GlobalAttentionModule(mlp2[-1], 
                additional_dim=3, attention_bn=global_attention_setting['attention_bn'],
                last_activation=global_attention_setting['last_activation'])

    def forward(self, unknown, known, unknow_feats, known_feats, t_emb=None, condition_emb=None, 
                second_condition_emb=None, record_neighbor_stats=False, pooling='max'):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        # known_feats are features at positions specified at known
        # unknow_feats are features at positions specified at unknown
        # unknow_feats features at the positons unknown obtained in the encoder

        # we first interpolate known_feats to interpolated_feats at positions unknown
        # then concat interpolated_feats with unknow_feats to get new_features
        # then new_features is put through a pointnet 
        if self.use_attention_module or self.use_global_attention_module:
            assert known is not None
            assert unknown is not None
            if self.use_global_attention_module:
                assert not self.include_grouper
        if known is not None:
            grouped_feats = pointnet2_utils.group_knn(unknown, known, known_feats, self.K, transpose=True)
            # grouped_feats is of shape (B,C2+11,n,K)
            second_condition_emb = second_condition_emb if self.include_second_condition else None
            grouped_feats_out = self.mlp1(grouped_feats, t_emb=None, condition_emb=second_condition_emb)
            # grouped_feats_out is of shape (B, mlp1[-1],n,K)
            if self.use_attention_module:
                # unknow_feats (B, C1, n)
                # unknow_feats is used as query
                interpolated_feats = self.attention_module(unknow_feats, grouped_feats, grouped_feats_out, count='all')
                # (B, mlp1[-1], n)
            else:
                interpolated_feats = pooling_features(grouped_feats_out, count='all', pooling=pooling)
                # (B, mlp1[-1], n)
            
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1) # (B, mlp1[-1] + C1, n)
        else:
            new_features = interpolated_feats

        if self.include_grouper:
            new_features, count = self.grouper(unknown, unknown, new_features, subset=True, 
                                record_neighbor_stats=record_neighbor_stats, return_counts=True)
            # (B, mlp1[-1] + C1 + 3 + 3, n)
        else:
            new_features = torch.cat([new_features, unknown.transpose(1,2)], dim=1) # (B, mlp1[-1] + C1 + 3, n)
            new_features = new_features.unsqueeze(-1) # (B, mlp[-1] + C1 + 3, n, 1)
        
        
        t_emb = t_emb if self.include_t else None
        condition_emb = condition_emb if self.include_condition else None
        new_features = self.mlp2(new_features, t_emb=t_emb, condition_emb=condition_emb)
        
        if self.include_grouper:
            # new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  
            new_features = pooling_features(new_features, count=count, pooling=pooling)
            # shape (B, mlp2[-1], n)
            return new_features

        
        new_features = new_features.squeeze(-1) # shape (B, mlp2[-1], n)
        if self.use_global_attention_module:
            new_features = torch.cat([new_features, unknown.transpose(1,2)], dim=1) # (B, mlp2[-1] + 3, n)
            new_features = self.global_attention_module(new_features)
        return new_features


if __name__ == '__main__':
    import pdb
    torch.cuda.set_device(2)

    B = 4
    N = 2048
    C = 256 # input feature channel
    activation = 'swish'
    attention_setting={'use_attention_module':True, 'attention_bn': True, 
                    'transform_grouped_feat_out':True, 'last_activation':True}
    global_attention_setting = {'use_global_attention_module':True, 'attention_bn': True, 
                    'last_activation':True}
    # mlp_spec = [32, 64, 64, 128]
    # bn = True
    t_dim=128
    condition_dim=128
    second_condition_dim=1024
    include_t=True
    include_condition=True
    include_second_condition=True

    xyz = torch.rand(B, N, 3).cuda() 
    features = torch.rand(B, C, N).cuda()
    t_emb = torch.rand(B, t_dim).cuda()
    condition_emb = torch.rand(B, condition_dim).cuda()
    second_condition_emb = torch.rand(B, second_condition_dim).cuda()
    
    sa = PointnetSAModule(
            npoint=64,
            radius=0.2,
            nsample=32,
            mlp=[32, 64, 64, 128],
            use_xyz=True, bn=True,
            t_dim=t_dim, include_t=include_t,
            include_abs_coordinate=True,
            include_center_coordinate=True,
            bn_first=True, bias=True, first_conv=True,
            first_conv_in_channel=C, res_connect=True,
            include_condition=include_condition, condition_dim=condition_dim,
            include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
            neighbor_def='radius', activation=activation, attention_setting=attention_setting,
            global_attention_setting=global_attention_setting)
    
    sa.cuda()
    new_xyz, new_features = sa(xyz, features, t_emb, condition_emb, second_condition_emb, 
                    subset=True, record_neighbor_stats=True)
    print('SA module:', sa)
    pdb.set_trace()

    xyz = torch.rand(B, N, 3).cuda() 
    features = torch.rand(B, C, N).cuda()
    npoint = 512
    new_xyz = torch.rand(B, npoint, 3).cuda() 
    query_feature_dim = 60
    new_xyz_feature = torch.rand(B, query_feature_dim, npoint).cuda()
    attention_setting={'use_attention_module':True, 'attention_bn': True, 
                    'transform_grouped_feat_out':True, 'last_activation':True}
    fp = FeatureMapModule([32, 64, 64, 128], radius=0.2, K=32, use_xyz=True, include_abs_coordinate=True, include_center_coordinate=True,
                    bn=True, bn_first=True, bias=True, res_connect=True,
                    first_conv=True, first_conv_in_channel=C, neighbor_def = 'radius', activation='relu', attention_setting=attention_setting,
                    query_feature_dim=query_feature_dim)
    fp.cuda()
    
    out_features = fp(xyz, features, new_xyz, subset=False, record_neighbor_stats=False, pooling='max',
                    features_at_new_xyz=new_xyz_feature)
    print(fp)
    pdb.set_trace()

    
    n = 512
    m = 256
    C2 = 128
    C1 = 64

    known = torch.rand(B, m, 3).cuda()
    known_feats = torch.rand(B, C2, m).cuda()
    unknown = torch.rand(B, n, 3).cuda()
    unknow_feats = torch.rand(B, C1, n).cuda()
    fp = PointnetFPModule([64, 64, 64, 32], bn=True, t_dim=t_dim, include_t=include_t,
                    bn_first=True, bias=True, first_conv=True,
                    first_conv_in_channel=C1+C2, res_connect=True,
                    include_condition=include_condition, condition_dim=condition_dim,
                    include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                    include_grouper = False,
                    radius=0.2, nsample=32, use_xyz=True, 
                    include_abs_coordinate=True,
                    include_center_coordinate=True,
                    neighbor_def='radius', activation=activation)
    
    fp.cuda()
    print('FP module', fp)

    new_fea = fp(unknown, known, unknow_feats, known_feats, t_emb=t_emb, condition_emb=condition_emb, 
                    second_condition_emb=second_condition_emb)
    # pdb.set_trace()
    # new_fea is of shape (B, mlp[-1], n) 
    K = 8
    attention_setting={'use_attention_module':True, 'attention_bn': True, 
                    'transform_grouped_feat_out':True, 'last_activation':True}
    global_attention_setting = {'use_global_attention_module':True, 'attention_bn': True, 
                    'last_activation':True}
    knn_fp = PointnetKnnFPModule([C2,C2,C2], [C1+C2, 64, 32], K, bn=True, t_dim=t_dim, include_t=include_t,
                    bn_first=True, bias=True, first_conv=False,
                    first_conv_in_channel1=C2, first_conv_in_channel2=C1+C2, res_connect=True,
                    include_condition=include_condition, condition_dim=condition_dim,
                    include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                    include_grouper = False,
                    radius=0.2, nsample=32, use_xyz=True, 
                    include_abs_coordinate=True,
                    include_center_coordinate=True,
                    neighbor_def='radius', activation=activation,
                    attention_setting=attention_setting,
                    global_attention_setting=global_attention_setting)
    knn_fp.cuda()
    print('Knn FP module', knn_fp)
    knn_fea = knn_fp(unknown, known, unknow_feats, known_feats, t_emb=t_emb, condition_emb=condition_emb,
                    second_condition_emb=second_condition_emb)
    # knn_fea is of shape (B, mlp[-1], n) 
    pdb.set_trace()
    