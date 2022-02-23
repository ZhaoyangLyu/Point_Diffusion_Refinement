import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from pointnet2.models.pnet import Pnet2Stage
# import numpy as np

# from .common import *
class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet(Module):

    def __init__(self, residual=True,
                    num_steps=1000, beta_1=1e-4, beta_T=0.05, mode='linear',
                    pnet_global_feature_architecture=[[3, 128, 256],[512, 1024]],
                    global_feature_remove_last_activation=False):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.var = VarianceSchedule(num_steps, beta_1, beta_T, mode='linear')
        context_dim = pnet_global_feature_architecture[1][-1]
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

        self.global_feature_dim = pnet_global_feature_architecture[1][-1]
        
        self.global_pnet = Pnet2Stage(pnet_global_feature_architecture[0],
                                        pnet_global_feature_architecture[1],
                                        bn=False, remove_last_activation=global_feature_remove_last_activation)

    def forward(self, x, condition, ts, label=None, use_retained_condition_feature=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            ts:     Time. (B, ).
            condition:  condition point cloud. (B, M, d2).
        """
        if ts is None:
            ts = torch.zeros(x.shape[0], device=x.device)
        ts = ts.long()
        batch_size = x.size(0)
        beta = self.var.betas[ts] # (B,)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = self.global_pnet(condition.transpose(1,2))
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out

    def reset_cond_features(self):
        return 0

def get_pointwise_net(args):
    net = PointwiseNet(**args)
    return net

if __name__ == '__main__':
    import pdb
    # point_dim = 3
    # context_dim = 1024
    # residual = True
    # net = PointwiseNet(residual, pnet_global_feature_architecture=[[4, 128, 256],[512, 1024]])
    network_args={
            "residual": True,
            "num_steps": 1000, 
            "beta_1": 1e-4, 
            "beta_T": 0.05, 
            "mode": "linear",
            "pnet_global_feature_architecture": [[4, 128, 256],[512, 1024]],
            "global_feature_remove_last_activation": False
        }
    net = get_pointwise_net(network_args)
    print(net)

    # num_steps = 1000
    # beta_1 = 1e-4
    # beta_T = 0.05
    # var = VarianceSchedule(num_steps, beta_1, beta_T, mode='linear')

    B = 32
    N = 2048
    x = torch.rand(B, N, 3)
    T = torch.randint(1000, size=(B,))
    # beta = var.betas[T]
    # context = torch.rand(B, context_dim)
    condition = torch.rand(B, 3072, 4)
    out = net(x, condition, T)

    # pdb.set_trace()

    

    # T = torch.randint(1000, size=(B,))

    pdb.set_trace()
    



