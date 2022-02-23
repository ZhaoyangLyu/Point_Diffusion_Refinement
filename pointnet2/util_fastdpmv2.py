import os
import numpy as np
import torch
import pickle
import pdb

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
    # return torch.normal(0, 1, size=size)
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


""" Fast DPM sampling"""
# https://github.com/FengNiMa/FastDPM_pytorch

def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value
    
    Returns:
    x (float)
    """
    # 
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2 
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(S, diffusion_config, schedule='linear'):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of noise levels, size = (S, )
    """
    target = np.prod(1 - np.linspace(diffusion_config["beta_0"], diffusion_config["beta_T"], diffusion_config["T"]))

    if schedule == 'linear':
        g = lambda x: np.linspace(diffusion_config["beta_0"], x, S)
        domain = (diffusion_config["beta_0"], 0.99)
    elif schedule == 'quadratic':
        g = lambda x: np.array([diffusion_config["beta_0"] * (1+i*x) ** 2 for i in range(S)])
        domain = (0.0, 0.95 / np.sqrt(diffusion_config["beta_0"]) / S)
    else:
        raise NotImplementedError

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def get_STEP_step(S, diffusion_config, schedule='linear'):
    """
    Compute STEP steps

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of steps, size = (S, )
    """
    if schedule == 'linear':
        c = (diffusion_config["T"] - 1.0) / (S - 1.0)
        list_tau = [np.floor(i * c) for i in range(S)]
    elif schedule == 'quadratic':
        list_tau = np.linspace(0, np.sqrt(diffusion_config["T"] * 0.8), S) ** 2
    else:
        raise NotImplementedError

    return [int(s) for s in list_tau]


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    Alpha_bar = Alpha_bar.cuda()

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    # Beta_tilde = torch.from_numpy(user_defined_eta).to(DEVICE).to(torch.float32)
    Beta_tilde = torch.from_numpy(user_defined_eta).cuda().to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user-1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i+1]:
                    t_adapted = bisearch(f=lambda _t: _log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T), 
                                            domain=(i-0.01, i+1.01), 
                                            target=np.log(Gamma_bar[t].cpu().numpy()))
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps


def VAR_sampling(net, size, diffusion_hyperparams,  # DDPM parameters
                user_defined_eta, kappa, continuous_steps,  # FastDPM parameters
                print_every_n_steps=100, label=0, verbose=True, condition=None):
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
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    # Beta_tilde = torch.from_numpy(user_defined_eta).to(DEVICE).to(torch.float32)
    Beta_tilde = torch.from_numpy(user_defined_eta).cuda().to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]
    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    print('begin sampling, total number of reverse steps = %s' % T_user)

    x = std_normal(size)
    # if not mean_shape is None:
    #     x = x + mean_shape * scale_factor # we assume mean_shape is in the scale of [-1,1]
    if not label is None and isinstance(label, int):
        # label = torch.ones(size[0]).long().to(DEVICE) * label
        label = torch.ones(size[0]).long().cuda() * label

    with torch.no_grad():
        for i, tau in enumerate(continuous_steps):
            if verbose:
                print('t %.2f x max %.2f min %.2f' % (tau, x.max(), x.min()))
            
            diffusion_steps = (tau * torch.ones((size[0],))).cuda()  # use the corresponding reverse step
            
            if condition is None:
                epsilon_theta = net(x, ts=diffusion_steps, label=label)  # predict \epsilon according to \epsilon_\theta
            else:
                epsilon_theta = net(x, condition, ts=diffusion_steps, label=label, use_retained_condition_feature=True)
            if verbose:
                print('t %.2f epsilon_theta max %.2f min %.2f' % (tau, epsilon_theta.max(), epsilon_theta.min()))
            
            if i == T_user - 1:  # the next step is to generate x_0
                assert abs(tau) < 0.1
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Gamma_bar[T_user-1-i - 1]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
            x *= torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
            x += c * epsilon_theta + sigma * std_normal(size)

            # if not mean_shape is None:
            #     x = x + (1-1/sqrt_Alpha) * mean_shape * scale_factor

    if not condition is None:
        net.reset_cond_features()

    return x


def STEP_sampling(net, size, diffusion_hyperparams,  # DDPM parameters
                user_defined_steps, kappa,  # FastDPM parameters
                print_every_n_steps=100, label=0, verbose=True, condition=None):
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
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined steps
    T_user = len(user_defined_steps)
    user_defined_steps = sorted(list(user_defined_steps), reverse=True)
    
    print('begin sampling, total number of reverse steps = %s' % T_user)

    x = std_normal(size)
    # if not mean_shape is None:
    #     x = x + mean_shape * scale_factor # we assume mean_shape is in the scale of [-1,1]
    if not label is None and isinstance(label, int):
        label = torch.ones(size[0]).long().cuda() * label

    with torch.no_grad():
        for i, tau in enumerate(user_defined_steps):
            if verbose:
                print('t %.2f x max %.2f min %.2f' % (tau, x.max(), x.min()))
            
            diffusion_steps = (tau * torch.ones(size[0])).cuda()  # use the corresponding reverse step
            
            if condition is None:
                epsilon_theta = net(x, ts=diffusion_steps, label=label)  # predict \epsilon according to \epsilon_\theta
            else:
                epsilon_theta = net(x, condition, ts=diffusion_steps, label=label, use_retained_condition_feature=True)
            if verbose:
                print('t %.2f epsilon_theta max %.2f min %.2f' % (tau, epsilon_theta.max(), epsilon_theta.min()))
            
            if i == T_user - 1:  # the next step is to generate x_0
                assert tau == 0
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Alpha_bar[user_defined_steps[i+1]]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next))
            x *= torch.sqrt(alpha_next / Alpha_bar[tau])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Alpha_bar[tau]) * torch.sqrt(alpha_next / Alpha_bar[tau])
            x += c * epsilon_theta + sigma * std_normal(size)

            # if not mean_shape is None:
            #     x = x + (1-1/sqrt_Alpha) * mean_shape * scale_factor

    if not condition is None:
        net.reset_cond_features()

    return x


def fast_sampling_function_v2(net, size, diffusion_hyperparams, diffusion_config, # DDPM parameters
                length = 100, sampling_method = 'var',  schedule = 'quadratic', kappa = 0.0,
                print_every_n_steps=100, label=0, verbose=True, condition=None):
    assert sampling_method in ['var', 'step']
    assert schedule in ['quadratic', 'linear']

    if sampling_method == 'var':
        user_defined_eta = get_VAR_noise(length, diffusion_config, schedule)
        continuous_steps = _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta)
        X = VAR_sampling(net, size, diffusion_hyperparams, 
                        user_defined_eta, kappa, continuous_steps,
                        print_every_n_steps=print_every_n_steps, label=label, 
                        verbose=verbose, condition=condition)
        return X
    else:
        user_defined_steps = get_STEP_step(length, diffusion_config, schedule)
        # print("Discrete steps:", user_defined_steps)
        X = STEP_sampling(net, size, diffusion_hyperparams, 
                        user_defined_steps, kappa, 
                        print_every_n_steps=print_every_n_steps, label=label, 
                        verbose=verbose, condition=condition)
        return X


if __name__ == '__main__':
    # tensor shape checker
    net = lambda x, ts, label: x
    diffusion_config = {"T": 1000, "beta_0": 0.0001, "beta_T": 0.02}
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    length = 10
    size = (16, 64, 3)

    for sampling_method in ['var', 'step']:
        for schedule in ["linear", "quadratic"]:
            for kappa in [0.0, 0.2, 0.5, 1.0]:
                X = fast_sampling_function_v2(net, size, diffusion_hyperparams, diffusion_config, # DDPM parameters
                        length = length, sampling_method = sampling_method,  schedule = schedule, kappa = kappa,
                        print_every_n_steps=100, label=None, scale_factor=1, 
                        mean_shape=None, verbose=False, condition=None)
                print(X.shape)