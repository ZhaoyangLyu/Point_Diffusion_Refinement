import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetKnnFPModule
from torch.utils.data import DataLoader


import copy

def swish(x):
    return x * torch.sigmoid(x)

import numpy as np

def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    # input is of shape (B) of integer time steps
    # output is of shape (B, t_emb_dim)
    ts = ts.unsqueeze(1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device) # shape (half_dim)
    # ts is of shape (B,1)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)
    
    return t_emb

# class PointNet2SemSegSSG(PointNet2ClassificationSSG):
class PointNet2SemSegSSG(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._build_model()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def build_SA_model(self, npoint, radius, nsample, feature_dim, mlp_depth, in_fea_dim, 
                            include_t, include_class_condition, class_condition_dim=None,
                            include_global_feature=False, global_feature_dim=None,
                            additional_fea_dim=None, neighbor_def='radius', activation='relu',
                            bn=True, attention_setting=None, global_attention_setting=None):
        SA_modules = nn.ModuleList()
        # include_t = self.hparams['include_t']
        if not isinstance(neighbor_def, list):
            neighbor_def = [neighbor_def] * len(radius)
        t_dim = self.hparams['t_dim']
        # pdb.set_trace()
        for i in range(len(npoint)):
            if additional_fea_dim is None:
                mlp_spec = [feature_dim[i]]*mlp_depth + [feature_dim[i+1]]
            else:
                # mlp_spec = [ feature_dim[i]+additional_fea_dim[i] ]*mlp_depth + [feature_dim[i+1]]
                mlp_spec = [feature_dim[i]]*mlp_depth + [feature_dim[i+1]]
                mlp_spec[0] = mlp_spec[0] + additional_fea_dim[i]
            
            first_conv=self.hparams["bn_first"]
            if i==0:
                # if bn_first, we will need to add a conv layer at the beginning
                # first_mlp_input_dim = mlp_spec[1] if first_conv else in_fea_dim
                if not first_conv:
                    mlp_spec[0] = in_fea_dim
            else:
                first_conv = False
            
            if include_global_feature:
                include_condition = True
                condition_dim = global_feature_dim
                include_second_condition = include_class_condition
                second_condition_dim = self.hparams["class_condition_dim"] if (class_condition_dim is None) else class_condition_dim
            else:
                include_condition = include_class_condition
                condition_dim = self.hparams["class_condition_dim"] if (class_condition_dim is None) else class_condition_dim
                include_second_condition = False
                second_condition_dim = None

            use_global_attention = ((not global_attention_setting is None) and 
                                    global_attention_setting['use_global_attention_module'] and
                                    i in global_attention_setting['global_attention_layer_index'])
            this_global_attention_setting = global_attention_setting if use_global_attention else None

            SA_modules.append(
                PointnetSAModule(
                    npoint=npoint[i],radius=radius[i],nsample=nsample[i],mlp=mlp_spec,use_xyz=self.hparams["model.use_xyz"],
                    t_dim=4*t_dim, include_t=include_t,include_abs_coordinate=self.include_abs_coordinate,
                    include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                    bn_first=self.hparams["bn_first"], first_conv=first_conv, first_conv_in_channel=in_fea_dim,
                    res_connect=self.hparams["res_connect"], bias = self.hparams["bias"],
                    include_condition=include_condition, condition_dim=condition_dim,
                    include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                    neighbor_def = neighbor_def[i], activation=activation, bn=bn,
                    attention_setting=attention_setting,
                    global_attention_setting=this_global_attention_setting
                ))
        return SA_modules

    def build_FP_model(self, decoder_feature_dim, decoder_mlp_depth, feature_dim, in_fea_dim, include_t, 
                            include_class_condition, class_condition_dim=None,
                            include_global_feature=False, global_feature_dim=None,
                            additional_fea_dim=None, use_knn_FP=False, K=3, 
                            include_grouper = False, radius=[0], nsample=[32], neighbor_def='radius',
                            activation = 'relu', bn=True, attention_setting=None,
                            global_attention_setting=None):
        FP_modules = nn.ModuleList()
        # include_t = self.hparams['include_t']
        t_dim = self.hparams['t_dim']
        if not isinstance(neighbor_def, list):
            neighbor_def = [neighbor_def] * len(radius)
        for i in range(len(decoder_feature_dim)-1):
            if i==0:
                skip_feature_dim = in_fea_dim
            else:
                skip_feature_dim = feature_dim[i]

            if include_global_feature:
                include_condition = True
                condition_dim = global_feature_dim
                include_second_condition = include_class_condition
                second_condition_dim = self.hparams["class_condition_dim"] if (class_condition_dim is None) else class_condition_dim
            else:
                include_condition = include_class_condition
                condition_dim = self.hparams["class_condition_dim"] if (class_condition_dim is None) else class_condition_dim
                include_second_condition = False
                second_condition_dim = None

            if use_knn_FP:
                # mlp1_spec = [decoder_feature_dim[i+1]] * (decoder_mlp_depth+1)
                # mlp2_spec = [decoder_feature_dim[i+1] + skip_feature_dim] + [decoder_feature_dim[i]] * decoder_mlp_depth
                mlp1_spec = [decoder_feature_dim[i+1]] + [decoder_feature_dim[i]] * decoder_mlp_depth
                mlp2_spec = [decoder_feature_dim[i] + skip_feature_dim] + [decoder_feature_dim[i]] * decoder_mlp_depth
                if not additional_fea_dim is None:
                    mlp1_spec[0] = mlp1_spec[0] + additional_fea_dim[i]
                    
                use_global_attention = ((not global_attention_setting is None) and 
                                    global_attention_setting['use_global_attention_module'] and
                                    i in global_attention_setting['global_attention_layer_index'])
                this_global_attention_setting = global_attention_setting if use_global_attention else None
                
                FP_modules.append(PointnetKnnFPModule(mlp1=mlp1_spec, mlp2=mlp2_spec, K=K, first_conv=False, bn=bn,
                                                    t_dim=4*t_dim, include_t=include_t, bn_first=self.hparams["bn_first"],
                                                    res_connect=self.hparams["res_connect"], bias = self.hparams["bias"],
                                                    include_condition=include_condition, condition_dim=condition_dim,
                            include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                                                    include_grouper=include_grouper, radius=radius[i], nsample=nsample[i], 
                                                    use_xyz=self.hparams["model.use_xyz"], 
                                                    include_abs_coordinate=self.include_abs_coordinate,
                                                    include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                                                    neighbor_def=neighbor_def[i], activation=activation,
                                                    attention_setting=attention_setting,
                                                    global_attention_setting=this_global_attention_setting))
            else:
                # if additional_fea_dim is None:
                mlp_spec = [decoder_feature_dim[i+1] + skip_feature_dim] + [decoder_feature_dim[i]] * decoder_mlp_depth
                # else:
                #     mlp_spec = [decoder_feature_dim[i+1] + skip_feature_dim + additional_fea_dim[i]] + [decoder_feature_dim[i]] * decoder_mlp_depth
                if not additional_fea_dim is None:
                    mlp_spec[0] = mlp_spec[0] + additional_fea_dim[i]
                FP_modules.append(PointnetFPModule(mlp=mlp_spec, first_conv=False, bn=bn,
                                                    t_dim=4*t_dim, include_t=include_t, bn_first=self.hparams["bn_first"],
                                                    res_connect=self.hparams["res_connect"], bias = self.hparams["bias"],
                                                    include_condition=include_condition, condition_dim=condition_dim,
                            include_second_condition=include_second_condition, second_condition_dim=second_condition_dim,
                                                    include_grouper=include_grouper, radius=radius[i], nsample=nsample[i], 
                                                    use_xyz=self.hparams["model.use_xyz"], 
                                                    include_abs_coordinate=self.include_abs_coordinate,
                                                    include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                                                    neighbor_def=neighbor_def[i], activation=activation))
        return FP_modules

    def _build_model(self):
        self.record_neighbor_stats = self.hparams["record_neighbor_stats"]
        self.scale_factor = self.hparams["scale_factor"]
        if self.hparams["include_class_condition"]:
            self.class_emb = nn.Embedding(self.hparams["num_class"], self.hparams["class_condition_dim"])
        self.attach_position_to_input_feature = self.hparams['attach_position_to_input_feature']
        in_fea_dim = self.hparams['in_fea_dim']
        if self.attach_position_to_input_feature:
            in_fea_dim = in_fea_dim + 3
        self.include_abs_coordinate = self.hparams['include_abs_coordinate']

        include_t = self.hparams['include_t']
        t_dim = self.hparams['t_dim']

        # timestep embedding fc layers
        self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
        self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
        self.activation = swish

        arch = self.hparams['architecture']
        npoint = arch['npoint']#[1024, 256, 64, 16]
        radius = arch['radius']#[0.1, 0.2, 0.4, 0.8]
        nsample = arch['nsample']#[32, 32, 32, 32]
        feature_dim = arch['feature_dim']#[32, 64, 128, 256, 512]
        mlp_depth = arch['mlp_depth']#3
        self.SA_modules = self.build_SA_model(npoint, radius, 
                                nsample, feature_dim, mlp_depth, in_fea_dim,
                                self.hparams['include_t'], self.hparams["include_class_condition"])


        decoder_feature_dim = arch['decoder_feature_dim']#[128, 128, 256, 256, 512]
        decoder_mlp_depth = arch['decoder_mlp_depth']#3
        assert decoder_feature_dim[-1] == feature_dim[-1]

        self.use_knn_FP = self.hparams.get('use_knn_FP', False)
        self.K = self.hparams.get('K', 3)
        self.FP_modules = self.build_FP_model(decoder_feature_dim, decoder_mlp_depth, feature_dim, in_fea_dim,
                                                self.hparams['include_t'], self.hparams["include_class_condition"],
                                                use_knn_FP=self.use_knn_FP, K=self.K)

        last_conv_in_dim = decoder_feature_dim[0]
        if self.use_knn_FP:
            last_conv_in_dim = last_conv_in_dim + 3
        if self.hparams["bn_first"]:
            self.fc_lyaer = nn.Sequential(
                nn.ReLU(True),
                nn.Conv1d(last_conv_in_dim, self.hparams['out_dim'], kernel_size=1),
            )
        else:
            self.fc_lyaer = nn.Sequential(
                nn.Conv1d(last_conv_in_dim, 128, kernel_size=1, bias=self.hparams["bias"]),
                # nn.BatchNorm1d(128),
                nn.GroupNorm(32, 128),
                nn.ReLU(True),
                # nn.Dropout(0.5),
                nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
            )

    def forward(self, pointcloud, ts=None, label=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # pointcloud[:,:,0:3] = pointcloud[:,:,0:3] / self.scale_factor
        if self.attach_position_to_input_feature:
            xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
            pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)
            # in this case, the input pointcloud is of shape (B,N,C)
            # the output pointcloud is of shape (B,N,C+3)
            # we want the X not only as position, but also as input feature
        
        xyz, features = self._break_up_pc(pointcloud)
        xyz = xyz / self.scale_factor
        # if pointcloud is of shape BN3, then xyz=pointcloud, features=None
        # if pointcloud is of shape BN(3+C), then xyz is of shape BN3, features is of shape (B,C,N)

        if (not ts is None) and self.hparams['include_t']:
            t_emb = calc_t_emb(ts, self.hparams['t_dim'])
            t_emb = self.fc_t1(t_emb)
            t_emb = self.activation(t_emb)
            t_emb = self.fc_t2(t_emb)
            t_emb = self.activation(t_emb)
        else:
            t_emb = None

        if (not label is None) and self.hparams['include_class_condition']:
            # label should be 1D tensor of integers of shape (B)
            class_emb = self.class_emb(label) # shape (B, condition_emb_dim)
        else:
            class_emb = None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            # print(i)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], t_emb=t_emb, condition_emb=class_emb,
                                                    subset=True, record_neighbor_stats=self.record_neighbor_stats)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            # i from -1 to -len(self.FP_modules)
            # equivalent to i from len(self.SA_modules)-1 to 0
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i],
                t_emb = t_emb, condition_emb=class_emb
            )
        if self.use_knn_FP:
            # l_features[0] is of shape (B, decoder_feature_dim[0], N)
            intermediate = torch.cat([l_features[0], xyz.transpose(1,2)], dim=1)
        else:
            intermediate = l_features[0]
        out = self.fc_lyaer(intermediate)
        out = torch.transpose(out, 1,2)
        return out

    def report_SA_module_neighbor_stats(self, SA_module, module_name='SA_module'):
        with torch.no_grad():
            num_groupers_per_layer = len(SA_module[0].groupers)
            neigh_stats = [[]] * num_groupers_per_layer
            neigh_quantile = [[]] * num_groupers_per_layer
            for i in range(len(SA_module)):
                for k in range(num_groupers_per_layer):
                    neigh_stats[k].append(SA_module[i].groupers[k].neighbor_stats)
                    neigh_quantile[k].append(SA_module[i].groupers[k].neighbor_num_quantile)
            
            for k in range(num_groupers_per_layer):
                neigh_stats[k] = torch.stack(neigh_stats[k], dim=0)
                neigh_quantile[k] = torch.stack(neigh_quantile[k], dim=0)
        for k in range(num_groupers_per_layer):
            print('%s: neighbor number (min, mean, max) in the %d-th grouper' % (module_name, k))
            print(neigh_stats[k])
            print('%s: neighbor quantile (0-0.1-1) in the %d-th grouper' % (module_name, k))
            print(neigh_quantile[k])

    def report_FP_module_neighbor_stats(self, FP_module, module_name='FP_module'):
        if FP_module[0].include_grouper:
            with torch.no_grad():
                neigh_stats = []
                neigh_quantile = []
                for i in range(len(FP_module)):
                    neigh_stats.append(FP_module[i].grouper.neighbor_stats)
                    neigh_quantile.append(FP_module[i].grouper.neighbor_num_quantile)
                
                neigh_stats = torch.stack(neigh_stats, dim=0)
                neigh_quantile = torch.stack(neigh_quantile, dim=0)

            print('%s: neighbor number (min, mean, max)' % (module_name))
            print(neigh_stats)
            print('%s: neighbor quantile (0-0.1-1)' % (module_name))
            print(neigh_quantile)
        else:
            print('%s has no grouper' % module_name)

    def report_neighbor_stats(self):
        if not self.record_neighbor_stats:
            print('neighbor stats is not recorded')
            return
        self.report_SA_module_neighbor_stats(self.SA_modules, module_name='Input cloud SA_module')


import pdb
def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)
if __name__ == '__main__':
    # print(0)
    param = {}
    # in_fea_dim = 4
    param["model.use_xyz"] = True
    param['in_fea_dim'] = 0
    param['out_dim'] = 3
    param['include_t'] = True
    param['t_dim'] = 128
    param["attach_position_to_input_feature"] = True
    param["include_abs_coordinate"] = True

    param["bn_first"]=True 
    param["bias"]=True
    param["res_connect"]=True

    param["include_class_condition"]=False
    param["num_class"]=40
    param["class_condition_dim"]=128

    param["scale_factor"]=1 # the input will be divided by scale_factor

    param["architecture"] = {
            "npoint": [1024, 256, 64, 16],
            "radius": [0.1, 0.2, 0.4, 0.8],
            "nsample": [32, 32, 32, 32],
            "feature_dim": [32, 64, 128, 256, 512],
            "mlp_depth": 3,
            "decoder_feature_dim": [128, 128, 256, 256, 512],
            "decoder_mlp_depth": 2
        }

    param["use_knn_FP"] = True
    param["K"] = 3

    param["record_neighbor_stats"] = False

    device = torch.device('cuda:0')
    pnet_sem = PointNet2SemSegSSG(param)
    pnet_sem.to(device)
    print('pnet:', pnet_sem)
    print_size(pnet_sem)
    B = 64
    N = 2048
    # t_emb = torch.rand(B, param['t_dim']).to(device)
    # t_emb.requires_grad=True
    
    cloud = torch.rand(B, N, 3+param['in_fea_dim']).to(device)
    cloud.requires_grad=True
    ts = torch.randint(10, (B,)).to(device)
    label = torch.randint(param["num_class"], (B,)).to(device)
    out = pnet_sem(cloud, ts=ts, label=label)
    loss = out.mean()
    loss.backward()
    pdb.set_trace()