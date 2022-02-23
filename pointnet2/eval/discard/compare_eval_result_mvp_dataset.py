import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle

import pdb

def plot_result_list(result_list, plot_key, label_list, save_dir, title=None, suffix='', line_style=None, plot_values=None):
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
        
            plt.xlabel(plot_key, fontsize=40)
            plt.ylabel(key, fontsize=40)
            plt.title(title, fontsize=50)
            plt.legend(fontsize=40)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            save_file = os.path.join(save_dir, key+'_'+suffix+'.png')
            plt.savefig(save_file)
            plt.close()
            print('have save the figure for %s to the file %s' % (key, save_file))

    return 0

if __name__ == '__main__':
    path_list = [
    '../exp_mvp_dataset_completion/T1000_betaT0.02_shape_completion_include_class_condition_scale_1_no_random_replace_partail_with_complete',

    '../exp_mvp_dataset_completion/T1000_betaT0.02_shape_completion_no_class_condition_scale_0.5_no_random_replace_partail_with_complete',

    '../exp_mvp_dataset_completion/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_no_random_replace_partail_with_complete',

    '../exp_mvp_dataset_completion/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_random_replace_partail_with_complete_prob_0.3'
    ]

    label_list = ['include class scale 1', 'no class scale 0.5', 'no class scale 1', 'no class scale 1 rand gt 0.3']#, 'pnet radius mapper radius']


    file_list = [os.path.join(p, 'eval_result/gathered_eval_resulttrainset.pkl') for p in path_list]
    plot_values = ['avg_cd', 'avg_emd']
    result_list = []
    for f in file_list:
        handle = open(f, 'rb')
        result = pickle.load(handle)
        result_list.append(result)
        handle.close()
    plot_result_list(result_list, 'iter', label_list, './compare_mvp_dataset_completion', title='trainset', suffix='trainset',line_style=None, plot_values=plot_values)

    file_list = [os.path.join(p, 'eval_result/gathered_eval_result.pkl') for p in path_list]
    plot_values = ['avg_cd', 'avg_emd']
    result_list = []
    for f in file_list:
        handle = open(f, 'rb')
        result = pickle.load(handle)
        result_list.append(result)
        handle.close()
    plot_result_list(result_list, 'iter', label_list, './compare_mvp_dataset_completion', title='testset', suffix='testset',line_style=None, plot_values=plot_values)
    # pdb.set_trace()