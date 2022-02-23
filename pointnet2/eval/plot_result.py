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

def plot_result(result, plot_key, save_dir, plot_values=None, print_lowest_value=False):
    # result is a dictionary of lists
    # result[plot_key] is the horizontal axis
    # result[key] is vertical axis
    # we plot all other keys except plot_key against plot_key in result if plot_values is None
    # plot_values could aslo be a list of keys
    # we only plot those keys specified in plot_values against plot_key
    # print('\n Comparing current ckpt with previous saved ckpts', flush=True)
    os.makedirs(save_dir, exist_ok=True)
    x = np.array(result[plot_key])
    order = np.argsort(x)
    x = x[order]
    if len(result[plot_key]) > 1:
        for key in result.keys():
            plot = not key == plot_key
            if not plot_values is None:
                plot = plot and key in plot_values
            if plot:
                plt.figure()
                if isinstance(result[key], dict):
                    for sub_key in result[key].keys():
                        y = np.array(result[key][sub_key])
                        y = y[order]
                        plt.plot(x, y, marker = '.', label=sub_key)
                        if print_lowest_value:
                            find_and_print_lowest_value(x, y, plot_key, key+'-'+sub_key)
                    plt.xlabel(plot_key)
                    plt.legend()
                else:
                    y = np.array(result[key])
                    y = y[order]
                    plt.plot(x, y, marker = '.')
                    plt.xlabel(plot_key)
                    plt.ylabel(key)
                    if print_lowest_value:
                        find_and_print_lowest_value(x, y, plot_key, key)
                save_file = os.path.join(save_dir, key+'.png')
                plt.savefig(save_file)
                plt.close()
                print('have save the figure for %s to the file %s' % (key, save_file), flush=True)
    else:
        print('Do not plot because there is only 1 value in plot key', flush=True)
    return 0

if __name__ == '__main__':
    file_name = '../exp_shapenet/T1000_betaT0.02_shape_generation_noise_reduce_factor_5_corrected/eval_results/total_eval_result.pkl'
    handle = open(file_name, 'rb')
    result = pickle.load(handle)
    handle.close()

    plot_key = 'iter'
    save_dir = './'
    plot_result(result, plot_key, save_dir)
    # pdb.set_trace()