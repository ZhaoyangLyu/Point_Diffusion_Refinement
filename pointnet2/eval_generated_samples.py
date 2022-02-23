import os
import torch
import torch.nn as nn
import numpy as np
from mvp_dataloader.mvp_dataset import ShapeNetH5

try:
    from emd import EMD_distance
    EMD_module_loaded = True
except:
    print('The emd module is not loaded')
    EMD_module_loaded = False

from chamfer_loss_new import Chamfer_F1

import pdb

# from dataparallel import MyDataParallel


def evaluate(testloader, dataset='mvp_dataset', compute_emd=True, scale=0.5, parallel=True):
    
    metrics = {'cd_distance': torch.rand(0).cuda(), 'emd_distance': torch.rand(0).cuda(),
                'cd_p': torch.rand(0).cuda(), 'f1': torch.rand(0).cuda()}

    # cd_module = Chamfer_Loss()
    f1_threshold = 0.001 if dataset == 'mvp40' else 0.0001
    cd_module = Chamfer_F1(f1_threshold=f1_threshold)
    if compute_emd and EMD_module_loaded:
        emd_module = EMD_distance()

    if parallel:
        cd_module = nn.DataParallel(cd_module)
        if compute_emd and EMD_module_loaded:
            emd_module = nn.DataParallel(emd_module)

    for idx, data in enumerate(testloader):
        gt = data['complete'].cuda()
        generated = data['generated'].cuda()
        generated = generated/2/scale
        gt = gt/2/scale
        
        cd_p, dist, f1 = cd_module(generated, gt)
        if compute_emd and EMD_module_loaded:
            emd_cost = emd_module(generated, gt)
        else:
            emd_cost = torch.zeros_like(dist)

        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['emd_distance'] = torch.cat([metrics['emd_distance'], emd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])
        metrics['f1'] = torch.cat([metrics['f1'], f1])

    return metrics


if __name__ == "__main__":
    scale = 0.5
    parallel = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    batch_size = 256
    compute_emd = True

    dataset_name = 'mvp_dataset'
    generated_sample_path='generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_90_scale_1.2_translation_0.1/ckpt_643499/fast_sampling/fast_sampling_config_length_20_schedule_quadratic_kappa_0.5'
    dataset = ShapeNetH5('./mvp_dataloader/data/mvp_dataset', train=False, npoints=2048, novel_input=True, novel_input_only=False,
                            random_replace_partial_with_complete_prob=0, augmentation=False, scale=scale,
                            random_subsample=False, num_samples=1000,
                            use_a_random_indices_file=True,
                            random_indices_file = 'random_indices.pkl',
                            include_generated_samples=True, 
                            generated_sample_path=generated_sample_path,
                            use_mirrored_partial_input=False, number_partial_points=3072,
                            rank=0, world_size=1,
                            benchmark = False,
                            load_pre_computed_XT=False, T_step=10, 
                            XT_folder='data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2/ckpt_545999/',
                            append_samples_to_last_rank=False,
                            return_augmentation_params=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        metrics = evaluate(dataloader, dataset=dataset_name, compute_emd=compute_emd, scale=scale, parallel=parallel)

        for key in metrics.keys():
            print('The avg %s is %.8f' % (key, metrics[key].mean()))