import os
import numpy as np
import torch
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path, ckpt_name, mode='max', return_num_ckpts=False):
    """
    Find maximum epoch/iteration in path, formatted ${ckpt_name}_${n_iter}.pkl

    Parameters:
    path (str):         checkpoint path
    ckpt_name (str):    name of checkpoint
    mode (str): could be max, all, or best
        for best mode, we find the epoch with the lowest cd loss on test set
    
    Returns:
    maximum epoch/iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    # epoch = -1
    iterations = []
    for f in files:
        if len(f) <= len(ckpt_name) + 5:
            continue
        if f[:len(ckpt_name)] == ckpt_name and f[-4:]  == '.pkl' and ('best' not in f):
            number = f[len(ckpt_name)+1:-4]
            iterations.append(int(number))
    if return_num_ckpts:
        num_ckpts = len(iterations)
    if len(iterations) == 0:
        if return_num_ckpts:
            return -1, num_ckpts
        return -1
    if mode == 'max':
        if return_num_ckpts:
            return max(iterations), num_ckpts
        return max(iterations)
    elif mode == 'all':
        iterations = sorted(iterations, reverse=True)
        if return_num_ckpts:
            return iterations, num_ckpts
        return iterations
    elif mode == 'best':
        eval_file_name = os.path.join(path, '../../eval_result/gathered_eval_result.pkl')
        handle = open(eval_file_name, 'rb')
        data = pickle.load(handle)
        handle.close()
        cd = np.array(data['avg_cd'])
        idx = np.argmin(cd)
        itera = data['iter'][idx]
        print('We find iteration %d which has the lowest cd loss %.8f' % (itera, cd[idx]))
        if return_num_ckpts:
            return itera, num_ckpts
        return itera
    # elif mode == 'best_cd':
    #     for f in files:
    #         if len(f) <= len(ckpt_name) + 5:
    #             continue
    #         if f[:len(ckpt_name)] == ckpt_name and f[-4:]  == '.pkl' and ('best_cd' in f):

    else:
        raise Exception('%s mode is not supported' % mode)


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(diffusion_steps.device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), 
                                      torch.cos(_embed)), 1)
    
    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, print_every_n_steps=100, label=0,
                verbose=True, condition=None, return_multiple_t_slices=False,
                t_slices=[5, 10, 20, 50, 100, 200, 400, 600, 800],
                use_a_precomputed_XT=False, step=100, XT=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the pointnet model
    size (tuple):                   size of tensor to be generated
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    print_every_n_steps (int):      print status every this number of reverse steps          
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    
    print('begin sampling, total number of reverse steps = %s' % T)
    if return_multiple_t_slices:
        result_slices = {}
    x = std_normal(size)
    # if not mean_shape is None:
    #     x = x + mean_shape * scale_factor # we assume mean_shape is in the scale of [-1,1]
    if not label is None and isinstance(label, int):
        label = torch.ones(size[0]).long().cuda() * label
    if use_a_precomputed_XT:
        # assert mean_shape is None
        x = XT + Sigma[step] * std_normal(size)
        start_iter = step-1
    else:
        start_iter = T-1
    with torch.no_grad():
        for t in range(start_iter, -1, -1): # t from T-1 to 0
            if verbose:
                print('t%d x max %.2f min %.2f' % (t, x.max(), x.min()))
            if t % print_every_n_steps == 0:
                print('reverse step: %d' % t, flush=True)
            diffusion_steps = (t * torch.ones((size[0],))).cuda()  # use the corresponding reverse step
            # input_x = torch.cat([x,x], dim=2)
            # pdb.set_trace()
            # x= x/1.01
            # if (x.max()-x.min()) > 2 and T>100:
            #     x= x/1.01
            # pdb.set_trace()
            if condition is None:
                epsilon_theta = net(x, ts=diffusion_steps, label=label)  # predict \epsilon according to \epsilon_\theta
            else:
                epsilon_theta = net(x, condition, ts=diffusion_steps, label=label, use_retained_condition_feature=True)
            if verbose:
                print('t %d epsilon_theta max %.2f min %.2f' % (t, epsilon_theta.max(), epsilon_theta.min()))
            sqrt_Alpha = torch.sqrt(Alpha[t])
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / sqrt_Alpha  # update x_{t-1} to \mu_\theta(x_t)
            # if not mean_shape is None:
            #     x = x + (1-1/sqrt_Alpha) * mean_shape * scale_factor
            if return_multiple_t_slices and t in t_slices:
                result_slices[t] = x #.detach().cpu().numpy() # the slices are without noises
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
    if not condition is None:
        net.reset_cond_features()
    if return_multiple_t_slices:
        return x, result_slices
    else:
        return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, label=None, condition=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the pointnet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch cuda tensor (B,N,D)):  training data in batch
    mean shape is of shape (1, N, D)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    B, N, D = X.shape  # B is batchsize, N is number of points, D=3 is dimension
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    # diffusion_steps = torch.ones(B,1,1).long().cuda() * 999
    z = std_normal(X.shape) 
    sqrt_Alpha_bar = torch.sqrt(Alpha_bar[diffusion_steps])
    transformed_X = sqrt_Alpha_bar * X + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z   
    # if not mean_shape is None:
    #     transformed_X = transformed_X + (1-sqrt_Alpha_bar) * mean_shape
        # we assume X and mean_shape are in the scale of [-1,1]
    # compute x_t from q(x_t|x_0)
    # input_X = torch.cat([transformed_X, transformed_X], dim=2)
    if condition is None:
        epsilon_theta = net(transformed_X, ts=diffusion_steps.view(B,), label=label)  # predict \epsilon according to \epsilon_\theta
    else:
        epsilon_theta = net(transformed_X, condition, ts=diffusion_steps.view(B,), label=label)
    # net.report_neighbor_stats()
    # pdb.set_trace()
    return loss_fn(epsilon_theta, z)


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


import re
def find_config_file(file_name):
    if 'config' in file_name and '.json' in file_name:
        if os.path.isfile(file_name):
            return file_name
        else:
            print('The config file does not exist. Try to find other config files in the same directory')
            file_path = os.path.split(file_name)[0]
    else:
        if os.path.isdir(file_name):
            file_path = file_name
        else:
            raise Exception('%s does not exist' % file_name)
    # pdb.set_trace()
    files = os.listdir(file_path)
    files = [f for f in files if ('config' in f and '.json' in f)]
    print('We find config files: %s' % files)
    config = files[0]
    number = -1
    for f in files:
        all_numbers = re.findall(r'\d+', f)
        all_numbers = [int(n) for n in all_numbers]
        if len(all_numbers) == 0:
            this_number = -1
        else:
            this_number = max(all_numbers)
        if this_number > number:
            config = f
            number = this_number
    print('We choose the config:', config)
    return os.path.join(file_path, config)


import pdb
if __name__ == '__main__':
    
    # T = 1000
    # B = 32
    # embed_dim = 128
    # diffusion_steps = torch.randint(T, size=(B,1))
    # t1 = calc_diffusion_step_embedding(diffusion_steps, embed_dim)
    # t2 = calc_t_emb(diffusion_steps.view(B), embed_dim)
    file_name = './exp_shapenet/T1000_betaT0.02_shape_generation_noise_reduce_factor_10_corrected_emd_mean_shape/logs/checkpoint'
    config_file = find_config_file(file_name)
    print(config_file)
    pdb.set_trace()