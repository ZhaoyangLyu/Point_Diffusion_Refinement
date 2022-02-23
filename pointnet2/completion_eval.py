import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import pickle

from util import rescale, find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, AverageMeter
from util_fastdpmv2 import fast_sampling_function_v2

torch_version = torch.__version__
if torch_version == '1.7.1':
    from models.pointnet2_ssg_sem import PointNet2SemSegSSG
    from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
    from models.point_upsample_module import point_upsample
    from chamfer_loss_new import Chamfer_F1
    try:
        from emd import EMD_distance
        EMD_module_loaded = True
    except:
        print('The emd module is not loaded')
        EMD_module_loaded = False
elif torch_version == '1.4.0':
    import sys
    sys.path.append('models/pvd')
    from model_forward import PVCNN2
    from metrics.evaluation_metrics import EMD_CD
else:
    raise Exception('Pytorch version %s is not supported' % torch_version)

from dataset import get_dataloader



from eval.plot_result import plot_result
from eval.compare_eval_result import plot_result_list

import pdb

from dataparallel import MyDataParallel
import h5py
import time

name_to_number ={
'plane': '02691156',
'bench': '02828884',
'cabinet': '02933112',
'car': '02958343',
'chair': '03001627',
'monitor': '03211117',
'lamp': '03636649',
'speaker': '03691459',
'firearm': '04090263',
'couch': '04256520',
'table': '04379243',
'cellphone': '04401088',
'watercraft': '04530566'}

number_to_name = {}
for k in name_to_number.keys():
    number_to_name[name_to_number[k]] = k


def evaluate(net, testloader, diffusion_hyperparams, print_every_n_steps=200, parallel=True,
                dataset='shapenet', scale=1, save_generated_samples=False, save_dir = None,
                task = 'completion', refine_output_scale_factor=None, max_print_nums=1e8,
                save_multiple_t_slices=False,
                t_slices=[5, 10, 20, 50, 100, 200, 400, 600, 800],
                use_a_precomputed_XT=False, T_step=100,
                point_upsample_factor=1, include_displacement_center_to_final_output=False,
                compute_emd=True, compute_cd=True,
                num_points=None, augment_data_during_generation=False,
                noise_magnitude_added_to_gt=0.01, add_noise_to_generated_for_refine_exp=False,
                return_all_metrics=False,
                fast_sampling=False, fast_sampling_config=None, diffusion_config=None):
    assert task in ['completion', 'refine_completion', 'denoise']
    CD_meter = AverageMeter()
    F1_meter = AverageMeter()
    EMD_meter = AverageMeter()
    total_len = len(testloader)
    if fast_sampling:
        assert not save_multiple_t_slices
        assert not use_a_precomputed_XT

    if not dataset in ['shapenet', 'shapenet_pytorch', 'mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
         raise Exception('%s dataset is not supported' % dataset)
    if use_a_precomputed_XT:
        assert task == 'completion'
        assert dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet'] # right now we only implemented this feature for mvp dataset
    if augment_data_during_generation:
        assert task == 'completion'
        assert dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']
    if dataset == 'shapenet' or dataset=='shapenet_pytorch':
        total_meta = []
    elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        # total meta is label info
        total_meta = torch.rand(0).cuda().long()
    # cd_distance = torch.rand(0).cuda()
    # emd_distance = torch.rand(0).cuda()
    metrics = {'cd_distance': torch.rand(0).cuda(), 'emd_distance': torch.rand(0).cuda(),
                'cd_p': torch.rand(0).cuda(), 'f1': torch.rand(0).cuda()}

    # cd_module = Chamfer_Loss()
    f1_threshold = 0.001 if dataset == 'mvp40' else 0.0001
    if torch_version == '1.7.1':
        cd_module = Chamfer_F1(f1_threshold=f1_threshold)
        if compute_emd and EMD_module_loaded:
            emd_module = EMD_distance()

    if parallel:
        net = MyDataParallel(net)
        if torch_version == '1.7.1':
            cd_module = nn.DataParallel(cd_module)
            if compute_emd and EMD_module_loaded:
                emd_module = nn.DataParallel(emd_module)

    if save_generated_samples:
        print('generated_samples will be saved to the directory', save_dir)
        if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
            total_generated_data = None
            if save_multiple_t_slices:
                generated_data_t_slices = None

    print_interval = int(np.ceil(total_len / max_print_nums))
    total_time = 0
    for idx, data in enumerate(testloader):
        if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
            label = data['label'].cuda()
            condition = data['partial'].cuda()
            gt = data['complete'].cuda()
            if task == 'refine_completion':
                generated = data['generated'].cuda()
            if use_a_precomputed_XT:
                XT = data['XT'].cuda()
            else:
                XT = None
            if augment_data_during_generation:
                # in this case, condition, gt, generated, XT are all agumented
                M_inv = data['M_inv'].cuda()
                translation = data['translation'].cuda()
        
        batch = gt.shape[0]
        
        try:
            num_points = gt.shape[1]
        except:
            num_points = num_points
            print('num points is set to %d' % num_points)
            # print('num points is set to the number of points (%d) in the partial point cloud' % num_points)
        if (idx) % print_interval == 0:
            print('begin generating')
        net.reset_cond_features()

        start = time.time()
        # pdb.set_trace()
        if task == 'refine_completion':
            if add_noise_to_generated_for_refine_exp:
                generated = generated + torch.normal(0, noise_magnitude_added_to_gt, size=generated.shape, device=generated.device)
            displacement = net(generated, condition, ts=None, label=label)
            if point_upsample_factor > 1:
                generated_data, _ = point_upsample(generated, displacement, point_upsample_factor, 
                                                        include_displacement_center_to_final_output,
                                                        refine_output_scale_factor)
            else:
                generated_data = generated + displacement * refine_output_scale_factor
            # loss = loss_function(X, refined_X, batch_reduction='mean')
        elif task == 'denoise':
            generated = gt + torch.normal(0, noise_magnitude_added_to_gt, size=gt.shape, device=gt.device)
            displacement = net(generated, condition=condition, ts=None, label=label)
            generated_data = generated + displacement * refine_output_scale_factor
        else:
            if save_multiple_t_slices:
                assert dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']# shapenet is not supported yet
                generated_data, result_slices = sampling(net, (batch,num_points,3), 
                                diffusion_hyperparams, 
                                print_every_n_steps=print_every_n_steps, label=label, 
                                condition=condition,
                                verbose=False, return_multiple_t_slices=True,
                                t_slices=t_slices,
                                use_a_precomputed_XT=use_a_precomputed_XT, step=T_step, XT=XT)
                # result_slices is a dict that contains torch tensors
            else:
                if fast_sampling:
                    generated_data = fast_sampling_function_v2(net, (batch,num_points,3), diffusion_hyperparams,  # DDPM parameters
                                diffusion_config,
                                print_every_n_steps=print_every_n_steps, label=label, 
                                verbose=False, condition=condition,
                                **fast_sampling_config)
                else:
                    # generated_data = gt + torch.normal(0, 0.01, size=gt.shape, device=gt.device)
                    generated_data = sampling(net, (batch,num_points,3), 
                                diffusion_hyperparams, 
                                print_every_n_steps=print_every_n_steps, label=label, 
                                condition=condition,
                                verbose=False,
                                use_a_precomputed_XT=use_a_precomputed_XT, step=T_step, XT=XT)
        generation_time = time.time() - start
        total_time = total_time + generation_time
        # generated_data = torch.rand(batch,num_points,3,device=gt.device)
        if augment_data_during_generation:
            generated_data = torch.matmul(generated_data - translation, M_inv)
            gt = torch.matmul(gt - translation, M_inv)
        generated_data = generated_data/2/scale
        gt = gt/2/scale
        if save_multiple_t_slices:
            for key in result_slices.keys():
                if augment_data_during_generation:
                    result_slices[key] = torch.matmul(result_slices[key] - translation, M_inv)
                result_slices[key] = result_slices[key]/2/scale
                result_slices[key] = result_slices[key].detach().cpu().numpy()
                
        torch.cuda.empty_cache()
        
        if torch_version == '1.7.1':
            if compute_cd:
                cd_p, dist, f1 = cd_module(generated_data, gt)
                cd_loss = dist.mean().detach().cpu().item()
                f1_loss = f1.mean().detach().cpu().item()
            else:
                dist = torch.zeros(generated_data.shape[0], device=generated_data.device, dtype=generated_data.dtype)
                cd_p = dist
                f1 = dist
                cd_loss = dist.mean().detach().cpu().item()
                f1_loss = f1.mean().detach().cpu().item()

            if compute_emd and EMD_module_loaded:
                emd_cost = emd_module(generated_data, gt)
            else:
                emd_cost = torch.zeros_like(dist)
            emd_loss = emd_cost.mean().detach().cpu().item()
        else: # 1.4.0
            result = EMD_CD(generated_data, gt, f1_threshold = f1_threshold)
            dist = result['CD']
            cd_p = dist
            f1 = result['fscore']
            emd_cost = result['EMD']

            cd_loss = dist.mean().detach().cpu().item()
            f1_loss = f1.mean().detach().cpu().item()
            emd_loss = emd_cost.mean().detach().cpu().item()
            


        if dataset == 'shapenet':
            total_meta = total_meta + data[3]
        elif dataset == 'shapenet_pytorch':
            total_meta = total_meta + list(data[3])
        elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
            total_meta = torch.cat([total_meta, label])
        

        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['emd_distance'] = torch.cat([metrics['emd_distance'], emd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])
        metrics['f1'] = torch.cat([metrics['f1'], f1])

        CD_meter.update(cd_loss, n=batch)
        F1_meter.update(f1_loss, n=batch)
        EMD_meter.update(emd_loss, n=batch)
        if (idx) % print_interval == 0:
            print('progress [%d/%d] %.4f (%d samples) CD distance %.8f EMD distance %.8f F1 score %.6f this batch time %.2f total generation time %.2f' % (idx, total_len, 
                idx/total_len, batch, CD_meter.avg, EMD_meter.avg, F1_meter.avg, generation_time, total_time), flush=True)

        # if task == 'completion' and save_generated_samples:
        if save_generated_samples:
            if dataset in ['shapenet', 'shapenet_pytorch']:
                meta = data[3]
                # meta_files = [os.path.split(m)[-1] for m in meta]
                for i in range(len(meta)):
                    meta_split = meta[i].split('/')
                    meta_file = os.path.join(meta_split[-2], meta_split[-1])
                    save_file = os.path.join(save_dir, meta_file)
                    save_data = generated_data[i].detach().cpu().numpy()
                    hf = h5py.File(save_file, 'w')
                    hf.create_dataset('data', data=save_data)
                    hf.close()
            elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
                if dataset == 'mvp_dataset':
                    save_file = os.path.join(save_dir, 'mvp_generated_data_%dpts.h5' % num_points)
                elif dataset == 'shapenet_chunk':
                    save_file = os.path.join(save_dir, 'shapenet_generated_data_%dpts.h5' % num_points)
                elif dataset == 'mvp40':
                    save_file = os.path.join(save_dir, 'mvp40_generated_data_%dpts.h5' % num_points)
                elif dataset == 'partnet':
                    save_file = os.path.join(save_dir, 'partnet_generated_data_%dpts.h5' % num_points)
                if total_generated_data is None:
                    total_generated_data = generated_data.detach().cpu().numpy()
                else:
                    total_generated_data = np.concatenate([total_generated_data, 
                                            generated_data.detach().cpu().numpy()], axis=0)
                hf = h5py.File(save_file, 'w')
                hf.create_dataset('data', data=total_generated_data)
                hf.close()

                # save t slices
                if save_multiple_t_slices:
                    if generated_data_t_slices is None:
                        generated_data_t_slices = result_slices
                    else:
                        for t in t_slices:
                            generated_data_t_slices[t] = np.concatenate([generated_data_t_slices[t],
                                            result_slices[t]], axis=0)
                    
                    for t in t_slices:
                        if dataset == 'mvp_dataset':
                            t_save_file = os.path.join(save_dir, 'mvp_generated_data_%dpts_T%d.h5' % (num_points, t))
                        elif dataset == 'shapenet_chunk':
                            t_save_file = os.path.join(save_dir, 'shapenet_generated_data_%dpts_T%d.h5' % (num_points, t))
                        elif dataset == 'mvp40':
                            t_save_file = os.path.join(save_dir, 'mvp40_generated_data_%dpts_T%d.h5' % (num_points, t))
                        elif dataset == 'partnet':
                            t_save_file = os.path.join(save_dir, 'partnet_generated_data_%dpts_T%d.h5' % (num_points, t))
                        hf = h5py.File(t_save_file, 'w')
                        hf.create_dataset('data', data=generated_data_t_slices[t])
                        hf.close()

            if (idx) % print_interval == 0:
                print('%d files have been saved to the directory %s' % (batch, save_dir))
            

    if dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        total_meta = total_meta.detach().cpu().numpy()
    
    if return_all_metrics:
        return CD_meter.avg, EMD_meter.avg, total_meta, metrics
    else:
        return CD_meter.avg, EMD_meter.avg, total_meta, metrics['cd_distance'], metrics['emd_distance']

def get_each_category_distance(files):
    handle = open(files, 'rb')
    data = pickle.load(handle)
    handle.close()
    # pdb.set_trace()
    meta = data['meta']
    distance_keys = ['cd_distance', 'emd_distance']
    cate_split_result = []
    for distance in distance_keys:
        split_result = {}
        for k in name_to_number.keys():
            split_result[k] = []
        for i, m in enumerate(meta):
            number = m.split('/')[-2]
            cate = number_to_name[number]
            split_result[cate].append(data[distance][i])
        final_split_result = {}
        for k in split_result.keys():
            if len(split_result[k]) > 0:
                final_split_result[k] = np.array(split_result[k]).mean()
                # print(k, final_split_result[k])
        cate_split_result.append(final_split_result)
    for idx, dis in enumerate(distance_keys):
        new_key = dis + '_category_split_result'
        data[new_key] = cate_split_result[idx]
    handle = open(files, 'wb')
    pickle.dump(data, handle)
    handle.close()
    print('Have splitted distance of each category for file %s' % files, flush=True)
    return 0

def gather_eval_result_of_different_iters(directory, match1, match2, nomatch=None, split_category = False, save_suffix = '', plot=True,
    # gather all evaluation results from all ckpts and plot them in figures
    gathered_keys=['iter', 'avg_cd', 'avg_emd', 'cd_distance_category_split_result', 'emd_distance_category_split_result']):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files if match1 in f and match2 in f]
    if not nomatch is None:
        files = [f for f in files if not nomatch in f]
    gathered_results = {}
    for f in files:
        if split_category:
            get_each_category_distance(os.path.join(directory, f))
        handle = open(os.path.join(directory, f), 'rb')
        data = pickle.load(handle)
        handle.close()
        for key in gathered_keys:
            if key in data.keys():
                if isinstance(data[key], dict): # data[key] is a dictionary
                    if key in gathered_results.keys():
                        for sub_key in data[key].keys():
                            gathered_results[key][sub_key].append(data[key][sub_key])
                            # data[key][sub_key] is a single number
                    else:
                        gathered_results[key] = {}
                        for sub_key in data[key].keys():
                            gathered_results[key][sub_key] = [ data[key][sub_key] ]
                else: # data[key] is a single number
                    if key in gathered_results.keys():
                        gathered_results[key].append(data[key])
                    else:
                        gathered_results[key] = [data[key]]
            else:
                print('key %s is not in the data loaded from file %s' % (key, f), flush=True)
    save_file = os.path.join(directory, 'gathered_eval_result'+save_suffix+'.pkl')
    handle = open(save_file, 'wb')
    pickle.dump(gathered_results, handle)
    handle.close()
    if plot:
        plot_result(gathered_results, gathered_keys[0], os.path.join(directory, 'figures'+save_suffix), 
                    plot_values=gathered_keys[1:], print_lowest_value=False)
    return gathered_results

def plot_train_and_val_eval_result(eval_dir):
    # plot testset and trainset figures in the same figure, and find the ckpt that has the lowest loss value
    label_list = ['test set', 'train set']
    files = ['gathered_eval_result.pkl', 'gathered_eval_result_trainset.pkl']
    
    file_list = [os.path.join(eval_dir, files[i]) for i in range(len(files))]

    plot_values = ['avg_cd', 'avg_emd', 'avg_cd_p', 'avg_f1']
    result_list = []
    for f in file_list:
        handle = open(f, 'rb')
        result = pickle.load(handle)
        result_list.append(result)
        handle.close()

    save_dir = os.path.join(eval_dir, 'compare_test_and_train_set')
    plot_result_list(result_list, 'iter', label_list, save_dir, line_style=None, plot_values=plot_values,
                        print_lowest_value=True)


