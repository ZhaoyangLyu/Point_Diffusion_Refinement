import numpy as np
import pickle
import json
import torch
from evaluation_metrics import compute_all_metrics
from plot_result import plot_result

import os
import sys
sys.path.append('../')
from models.pointnet2_ssg_sem import PointNet2SemSegSSG
from util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, find_config_file
from dataset import get_dataloader
import argparse

def evaluate_gen(model, testloader, save_dir, ckpt_iter, num_points = 2048, batch_size=50):
    # device = torch.device('cuda:{}'.format(gpu))

    all_sample = []
    all_ref = []
    total_length = len(testloader)
    for iidx, data in enumerate(testloader):
        if train_config['dataset'] == 'shapenet':
            te_pc = data[0]
        elif train_config['dataset'] == 'pointflow_shapenet':
            te_pc = data['train_points']
        else:
            raise Exception(train_config['dataset'], 'dataset is not supported')
        te_pc = te_pc.cuda()
        B, N = te_pc.size(0), te_pc.size(1)
        # _, out_pc = model.sample(B, N)
        print('Generation progress: [%d/%d] %.4f' % (iidx, total_length, iidx/total_length), flush=True)
        out_pc = sampling(model, (B,num_points,3), diffusion_hyperparams, 
                            print_every_n_steps=diffusion_config["T"] // 5, label=None,
                            verbose=False)
        sys.stdout.flush()
        out_pc = out_pc.cuda()

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    del model
    torch.cuda.empty_cache()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s" % (sample_pcs.size(), ref_pcs.size()))

    # Save the generative output
    outfile = os.path.join(save_dir, "ckpt_%d_generated_point_cloud.pkl" % ckpt_iter)
    file_writer = open(outfile, 'wb')
    X = sample_pcs.cpu().detach().numpy()
    pickle.dump(X, file_writer)
    file_writer.close()
    print('saved generated samples at iteration %s to %s' % (ckpt_iter, outfile))

    # save ref points
    outfile = os.path.join(save_dir, "ref_point_cloud.pkl")
    file_writer = open(outfile, 'wb')
    X = ref_pcs.cpu().detach().numpy()
    pickle.dump(X, file_writer)
    file_writer.close()
    print('saved ref samples at iteration %s to %s' % (ckpt_iter, outfile))

    sys.stdout.flush()
    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=True)
    results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}
    results['iter'] = ckpt_iter
    # Save the computed metrics
    outfile = os.path.join(save_dir, "ckpt_%d_eval_result.pkl" % ckpt_iter)
    file_writer = open(outfile, 'wb')
    pickle.dump(results, file_writer)
    file_writer.close()
    print('saved computed metrics at iteration %s to %s' % (ckpt_iter, outfile))
    sys.stdout.flush()

    for key in results:
        print('{}\t{}'.format(key, results[key]))

    return results


if __name__ == '__main__':
    import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='all',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-b', '--batch_size', type=int, default=50,
                        help='batch size of the dataloader and sampling method')
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help='the gpu to use for evaluation')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_idx)
    print('Have set cuda visible devices to', args.gpu_idx)

    ckpt_iter=args.ckpt_iter
    batch_size = args.batch_size
    num_points = 2048
    with open(find_config_file(args.config)) as f:
        data = f.read()
    config = json.loads(data)
    print('The configuration is:')
    print(json.dumps(config, indent=4))


    gen_config  = config["gen_config"]
    global train_config
    train_config = config["train_config"]        # training parameters
    
    pointnet_config = config["pointnet_config"]     # to define pointnet
    global diffusion_config
    diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    # global trainset_config
    if train_config['dataset'] == 'modelnet40':
        trainset_config = config["trainset_config"]     # to load trainset
    elif train_config['dataset'] == 'shapenet':
        trainset_config = config["shapenet_config"]
        trainset_config['batch_size'] = batch_size
        trainset_config['data_dir'] = '../' + trainset_config['data_dir']
    elif train_config['dataset'] == 'pointflow_shapenet':
        trainset_config = config["pointflow_shapenet_config"]
        trainset_config['batch_size'] = batch_size
        trainset_config['data_dir'] = '../' + trainset_config['data_dir']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    global diffusion_hyperparams 
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)


    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    root_directory = '../' + train_config['root_directory']
    ckpt_path = gen_config['ckpt_path']
    ckpt_path = os.path.join(root_directory, local_path, ckpt_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path, 'pointnet_ckpt', mode='max')
    elif ckpt_iter == 'all':
        ckpt_iter = find_max_epoch(ckpt_path, 'pointnet_ckpt', mode='all')

    if not isinstance(ckpt_iter, list):
        ckpt_iter_list = [ckpt_iter]
    else:
        ckpt_iter_list = ckpt_iter

    print('Models at iterations %s will be evaluated' % ckpt_iter_list)

    ckpt_length = len(ckpt_iter_list)
    total_result = {}
    for idx, ckpt_iter in enumerate(ckpt_iter_list):
        print('testing %d-th [%d/%d, progress %.4f] checkpoint at iteration %d' % (idx, idx, ckpt_length, idx/ckpt_length, ckpt_iter))
        testloader = get_dataloader(trainset_config, phase='val')
        model_path = os.path.join(ckpt_path, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location='cpu')

        net = PointNet2SemSegSSG(pointnet_config).cuda()
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Model from %s at iter %s has been trained for %s seconds' % (model_path, ckpt_iter, checkpoint['training_time_seconds']))
        print_size(net)

        save_dir = os.path.join(root_directory, local_path, 'eval_results')
        print('Evaluation results will be saved to the directory', save_dir)
        os.makedirs(save_dir, exist_ok=True)

        sys.stdout.flush()
        with torch.no_grad():
            results = evaluate_gen(net, testloader, save_dir, ckpt_iter, num_points = 2048, batch_size=batch_size)
        if train_config['dataset'] == 'shapenet':
            testloader.kill_data_processes()
        
        for key in results.keys():
            if idx == 0:
                total_result[key] = [results[key]]
            else:
                total_result[key].append(results[key])
                plot_result(total_result, 'iter', os.path.join(save_dir, 'figures'))
        
        # save total_result
        outfile = os.path.join(save_dir, "total_eval_result.pkl")
        file_writer = open(outfile, 'wb')
        pickle.dump(total_result, file_writer)
        file_writer.close()
        print('saved total result up to iteration %s to %s' % (ckpt_iter, outfile))
        sys.stdout.flush()
    # pdb.set_trace()