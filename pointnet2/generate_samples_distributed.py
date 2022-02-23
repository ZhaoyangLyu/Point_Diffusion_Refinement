import argparse
import os
import pdb
import subprocess
from os import listdir
import h5py
import numpy as np
import pickle

def dict_to_command(dictionary, exclude_keys=[]):
    command = []
    for key in dictionary:
        if not key in exclude_keys:
            if isinstance(dictionary[key], bool):
                if dictionary[key]:
                    command.append('--' + str(key))
            else:
                command.append('--' + str(key))
                command.append(str(dictionary[key]))
    return command

def print_and_write(content, file_handle):
    print(content)
    file_handle.write(content + '\n')

def gather_generated_results(father_directory, num_ranks, remove_original_files=False):
    data = {}
    total_meta = [] 
    cd_distance = [] 
    emd_distance = []
    f1_score = []
    output_log_file = 'gathered_generation.log'
    output_log_file = os.path.join(father_directory, output_log_file)
    file_handle = open(output_log_file, 'w')
    for rank in range(num_ranks):
        directory = os.path.join(father_directory, 'rank_%d' % rank)
        files = listdir(directory)
        for fl in files:
            if fl[-3:] == '.h5': # generated coarse complete point clouds
                data_save_file = fl
                file_name = os.path.join(directory, fl)
                generated_file = h5py.File(file_name, 'r')
                generated_data = np.array(generated_file['data'])
                # data.append(generated_data)
                if data_save_file in data.keys():
                    data[data_save_file].append(generated_data)
                else:
                    data[data_save_file] = [generated_data]
                generated_file.close()
                print_and_write('data from %s is of shape %s' % (file_name, generated_data.shape), file_handle)
                if remove_original_files:
                    os.remove(file_name)
                    print_and_write('%s is removed' % (file_name), file_handle)
            elif fl[-4:] == '.pkl': # evaluation results
                eval_save_file = fl
                file_name = os.path.join(directory, fl)
                pkl_file_handle = open(file_name, 'rb')
                eval_result = pickle.load(pkl_file_handle)
                pkl_file_handle.close()
                # print('data from %s is of shape %s' % (file_name, generated_data.shape))
                total_meta.append(eval_result['meta'])
                cd_distance.append(eval_result['cd_distance'])
                emd_distance.append(eval_result['emd_distance'])
                f1_score.append(eval_result['f1'])
                iteration = eval_result['iter']
                if remove_original_files:
                    os.remove(file_name)
                    print_and_write('%s is removed' % (file_name), file_handle)


    
    # save the gathered generated data
    for key in data.keys():
        data[key] = np.concatenate(data[key], axis=0)
        print_and_write('The gathered data from all %s files of different ranks is of shape %s' % (key, data[key].shape), file_handle)
        save_dir = father_directory #os.path.split(directory)[0]
        gathered_data_save_file = os.path.join(save_dir, key)
        hf = h5py.File(gathered_data_save_file, 'w')
        hf.create_dataset('data', data=data[key])
        hf.close()
        print_and_write('The gathered data from all %s files of different ranks has been saved to %s' % (key, gathered_data_save_file), file_handle)

    # save the gathered eval results
    total_meta = np.concatenate(total_meta, axis=0)
    cd_distance = np.concatenate(cd_distance, axis=0)
    emd_distance = np.concatenate(emd_distance, axis=0)
    f1_score = np.concatenate(f1_score, axis=0)
    gathered_eval_save_file = os.path.join(save_dir, eval_save_file)
    handle = open(gathered_eval_save_file, 'wb')
    pickle.dump({'meta':total_meta, 'cd_distance':cd_distance, 
                    'emd_distance':emd_distance, 'f1':f1_score,
                    'avg_cd':cd_distance.mean(), 'avg_emd':emd_distance.mean(), 'iter':iteration}, handle)
    handle.close()
    print_and_write('have saved gathered eval result at iter %s to %s' % (str(iteration), gathered_eval_save_file), file_handle)
    print_and_write("CD loss: {} EMD loss: {} F1 Score: {}".format(cd_distance.mean(), emd_distance.mean(), f1_score.mean()), file_handle)

    file_handle.close()

if __name__ == "__main__":
    '''
    running examples:
    python generate_samples_distributed.py --execute --gather_results --remove_original_files --config exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json --ckpt_name pointnet_ckpt_643499.pkl --batch_size 32 --phase test --device_ids '0,1,2,3,4,5,6,7'
    
    python generate_samples_distributed.py --execute --gather_results --remove_original_files --config exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json --ckpt_name pointnet_ckpt_643499.pkl --batch_size 32 --phase test_trainset --save_multiple_t_slices --t_slices '[100]' --device_ids '0,1,2,3,4,5,6,7'
    
    python generate_samples_distributed.py --execute --gather_results --remove_original_files --config exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json --ckpt_name pointnet_ckpt_643499.pkl --batch_size 32 --phase test_trainset --use_a_precomputed_XT --T_step 100 --XT_folder mvp_dataloader/data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_90_scale_1.2_translation_0.1/pointnet_ckpt_643499 --augment_data_during_generation --augmentation_during_generation '1.2; 90; 0.5; 0.1' --num_trials 10 --device_ids '0,1,2,3,4,5,6,7'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                    help='if true, we run the generation jobs, if fasle, we only gather the generated results (default: false)')
    parser.add_argument('-g', '--gather_results', action='store_true',
                    help='whether to gather the generated results from different ranks (default: false)')
    parser.add_argument('--remove_original_files', action='store_true',
                    help='whether to romve original files from different ranks (default: false)')
    parser.add_argument('--num_ranks', type=int, default=-1,
                        help='number of ranks. We use num of gpus as num of ranks by default. We may want to use a larger num of ranks if we run exps on multi machines')
    parser.add_argument('--start_rank', type=int, default=0,
                        help='The starting evaluation rank. We may want to use a different start rank if we run exps on multi machines')

    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='number of points in each shape')
    parser.add_argument('--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max" or "best"')
    parser.add_argument('--ckpt_name', default='',
                        help='Which checkpoint to use, the file name of the ckeckpoint')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Number of data to be generated')
    parser.add_argument('-p', '--phase', type=str, default='test_trainset',
                        help='which part of the dataset to generated samples')
    parser.add_argument('--save_multiple_t_slices', action='store_true',
                    help='whether to save multiple t slices (default: false)')
    parser.add_argument('--t_slices', type=str, default='[5,10,20,50,100,200,400,600,800]',
                    help='the intermediate t slices to save')

    parser.add_argument('--fast_sampling', action='store_true',
                    help='whether to use fast sampling (default: false)')
    parser.add_argument('--fast_sampling_config', type=str, default='100; var; quadratic; 0.0',
                    help='fast_sampling_config: length; sampling_method; schedule type; kappa')

    parser.add_argument('--save_dir', type=str, default='',
                        help='the directory to save the generated samples')
    # parser.add_argument('--generate_samples_for_a_subset_of_the_datset', action='store_true',
    #                 help='whether to generate_samples_for_a_subset_of_the_datset (default: false)')
    # parser.add_argument('--subset_indices_file', type=str, default='mvp_dataloader/random_indices.pkl',
    #                     help='indices of the samples that we want to generate complete point clouds')

    parser.add_argument('--augment_data_during_generation', action='store_true',
                    help='whether to augment data during evaluation (default: false)')
    parser.add_argument('--augmentation_during_generation', type=str, default='1.2; 60; 0.5; 0.05',
                        help='augmentations during generation, (scale; rotation; mirror; translation)')

    parser.add_argument('--use_a_precomputed_XT', action='store_true',
                    help='whether to use precomputed XT to generate samples (default: false)')
    parser.add_argument('--T_step', type=int, default=100,
                        help='the t step to reverse begin with')
    # load_pre_computed_XT=False, T_step=100, XT_folder=None
    parser.add_argument('--XT_folder', type=str, default='mvp_dataloader/data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_90_scale_1.2_translation_0.1/pointnet_ckpt_643499',
                        help='the folder that stores the precomputed XT')

    
    parser.add_argument('-d', '--device_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='gpu device indices to use')
    parser.add_argument('-s', '--std_out_file', type=str, default='generation.log',
                        help='generation log output file')

    # multiple trials settings
    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of trials to generate for each partial point cloud')
    parser.add_argument('--start_trial', type=int, default=1,
                        help='trial index to start')
    args = parser.parse_args()

    exclude_keys = ['execute', 'gather_results', 'remove_original_files', 'num_ranks', 'start_rank']

    args_dict = vars(args)
    device_ids = args_dict['device_ids']
    device_ids = device_ids.split(',')
    num_gpus = len(device_ids)

    num_total_ranks = args.num_ranks if args.num_ranks > 0 else num_gpus
    start_rank = args.start_rank
    # be very careful to gather results when run exps on multi machines 

    if args.execute:
        workers = []
        for idx in range(num_gpus):
            args_dict['rank'] = idx + start_rank
            args_dict['world_size'] = num_total_ranks
            args_dict['device_ids'] = device_ids[idx]
            command = ['python', 'generate_samples.py']
            command = command + dict_to_command(args_dict, exclude_keys)
            print('%d-th command' % idx)
            print(command)

            p = subprocess.Popen(command)
            workers.append(p)

        for p in workers:
            p.wait()

        print('Have finished generating samples')
    
    if args.gather_results:
        if args.num_trials == 1:
            std_out_file = args.std_out_file.split('.')[0]
            std_out_file = std_out_file + '_rank_0.log'
            std_out_file = os.path.join('generation_logs', std_out_file)
            f = open(std_out_file, 'r')
            for line in f:
                # print(line)
                if 'generated_samples will be saved to the directory' in line:
                    break
            directory = line.split()[-1]
            father_directory = os.path.split(directory)[0]
            print('We will gather results from the directory', father_directory)
            gather_generated_results(father_directory, num_total_ranks, remove_original_files=args.remove_original_files)
        else:
            for trial_idx in range(args.start_trial, args.start_trial+args.num_trials):
                print('-' * 50)
                print('Gathering results for trial %d' % trial_idx)
                std_out_file = args.std_out_file.split('.')[0]
                std_out_file = std_out_file + '_trial_%d_rank_0.log' % trial_idx
                std_out_file = os.path.join('generation_logs', std_out_file)
                f = open(std_out_file, 'r')
                for line in f:
                    # print(line)
                    if 'generated_samples will be saved to the directory' in line:
                        break
                directory = line.split()[-1]
                father_directory = os.path.split(directory)[0]
                print('We will gather results from the directory', father_directory)
                gather_generated_results(father_directory, num_total_ranks, remove_original_files=args.remove_original_files)


        


