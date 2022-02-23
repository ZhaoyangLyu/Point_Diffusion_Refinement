import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle

import pdb

def find_and_print_lowest_value(x,y, x_key, y_key):
    idx = np.argmin(y)
    x_min = x[idx]
    y_min = y[idx]
    print('The lowest value of %s is %.8f at %s %.2f' % (y_key, y_min, x_key, x_min), flush=True)

def plot_result_list(result_list, plot_key, label_list, save_dir, line_style=None, plot_values=None, print_lowest_value=False):
    # result_list is a list of result
    # each result is a dictionary
    # label_list is a list of labels, it is of the same length of result_list
    os.makedirs(save_dir, exist_ok=True)

    for key in result_list[0].keys():
        plot = not key == plot_key
        if not plot_values is None:
            plot = plot and key in plot_values
        if plot:
        # if not key == plot_key:
            plt.figure(figsize=(15, 9))
            for idx, result in enumerate(result_list):
                x = np.array(result[plot_key])
                order = np.argsort(x)
                x = x[order]
                y = np.array(result[key])
                y = y[order]
                if line_style is None:
                    plt.plot(x, y, marker = '.', markersize=30, linewidth=5, label=label_list[idx])
                else:
                    plt.plot(x, y, line_style[idx], marker = '.', markersize=30, linewidth=5, label=label_list[idx])
                if print_lowest_value:
                    if key in ['avg_cd', 'avg_emd']:
                        find_and_print_lowest_value(x,y, plot_key, label_list[idx]+ '-' +key)
        
            plt.xlabel(plot_key, fontsize=40)
            plt.ylabel(key, fontsize=40)
            plt.legend(fontsize=40)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            save_file = os.path.join(save_dir, key+'.png')
            plt.savefig(save_file)
            plt.close()
            print('have saved the figure for %s to the file %s' % (key, save_file))

    return 0

if __name__ == '__main__':
    path_list = [
    '../exp_shapenet_completion/T1000_betaT0.02_shape_completion_include_global_feature_lr_0.0002_noise_reduce_factor_1_include_class_condition_False_augmentation_False/refine_exp_ckpt_934475_output_scale_0.001',

    '../exp_shapenet_completion/T1000_betaT0.02_shape_completion_include_global_feature_lr_0.0002_noise_reduce_factor_1_include_class_condition_False_augmentation_False/refine_exp_ckpt_934475_output_scale_0.001_avg_max_pooling',

    '../exp_shapenet_completion/T1000_betaT0.02_shape_completion_include_global_feature_lr_0.0002_noise_reduce_factor_1_include_class_condition_False_augmentation_False/refine_exp_ckpt_934475_output_scale_0.001_avg_max_pooling_swish_activation',

    '../exp_shapenet_completion/T1000_betaT0.02_shape_completion_include_global_feature_lr_0.0002_noise_reduce_factor_1_include_class_condition_False_augmentation_False/refine_exp_ckpt_934475_output_scale_0.001_fp_nn_16',

    '../exp_shapenet_completion/T1000_betaT0.02_shape_completion_include_global_feature_lr_0.0002_noise_reduce_factor_1_include_class_condition_False_augmentation_False/refine_exp_ckpt_934475_output_scale_0.001_fp_nn_32']# 
    # '../exp_shapenet_completion/exp_from_2037/T1000_betaT0.02_shape_completion_FP_grouper_True_use_knn_FP_True_K_3_pnet_nb_radius_mapper_nb_nn']

    label_list = ['max pool fp 8', 'avg max pool relu act', 'avg max pool swish act', 'max pool fp 16', 'max pool fp 32']#, 'pnet radius mapper radius']

    idx = [1,2]
    path_list = [path_list[i] for i in idx]
    label_list = [label_list[i] for i in idx]

    file_list = [os.path.join(p, 'eval_result/gathered_eval_resulttrainset.pkl') for p in path_list]
    plot_values = ['avg_cd', 'avg_emd']
    result_list = []
    for f in file_list:
        handle = open(f, 'rb')
        result = pickle.load(handle)
        result_list.append(result)
        handle.close()
    plot_result_list(result_list, 'iter', label_list, './compare_shapenet_completion', line_style=None, plot_values=plot_values)
    # pdb.set_trace()