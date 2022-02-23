import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='whether to execute the commands (default: False)')
    args = parser.parse_args()

    noise_reduce_factoruce = [1,3,5,10]
    devices = [4,5,6,7]
    batch_size = 81
    ckpt_suffix = 'logs/checkpoint'

    command = []
    for idx, factor in enumerate(noise_reduce_factoruce):
        exp_path = '../exp_pointflow_shapenet/pnet_with_knn_fp/airplane/T1000_betaT0.02_shape_generation_noise_reduce_factor_%d' % factor
        # if factor > 1:
        #     exp_path = exp_path + '_corrected'

        config_path = os.path.join(exp_path, ckpt_suffix)
        log_file = os.path.join(exp_path, 'eval.log')
        command.append('python generation_eval.py --config %s --batch_size %d --gpu_idx %s > %s &' % (
                        config_path, batch_size, devices[idx], log_file))

    for i in range(len(command)): 
        if args.execute:
            print('excuting %d-th command:' % i)
            print(command[i])
            os.system(command[i])
        else:
            print('%d-th command is:' % i)
            print(command[i])
        
