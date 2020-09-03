import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def denorm_conf(epoch_data_to_denorm):
    return [epoch_data_to_denorm[i] * epoch_data_to_denorm[8] for i in range(8)]


##############################################################################


def plot_one_head_conf_as_function_of_batch(module_type, head_layer,
                                            attn_type, head_num, epoch_data):
    num_of_batches_in_epoch = (epoch_data[module_type][head_layer][attn_type].shape[0])
    confs = []
    batches = []
    for batch in range(num_of_batches_in_epoch):
        batches.append(batch)
        epoch_data_tmp = epoch_data[module_type][head_layer][attn_type][batch]
        epoch_data_denormalized = denorm_conf(epoch_data_tmp)
        confs.append(epoch_data_tmp[head_num])
    plt.title("{} - {} Head {} Layer {} - Conf Along Batches".format(module_type, attn_type, head_num, head_layer))
    plt.xlabel("Batch")
    plt.ylabel("Confidence")
    plt.plot(batches, confs)


##############################################################################


def plot_layer_conf_as_function_of_batch(module_type, layer,
                                         attn_type, number_of_heads, epoch_data):
    num_of_batches_in_epoch = (epoch_data[module_type][0][attn_type].shape[0])
    confs = []
    batches = []
    for batch in range(num_of_batches_in_epoch):
        batches.append(batch)
        epoch_data_tmp = epoch_data[module_type][layer][attn_type][batch]
        epoch_data_denormalized = denorm_conf(epoch_data_tmp)
        confs.append(epoch_data_tmp)

    for num_head in range(number_of_heads):
        confs_of_head = [confs[b][num_head] for b in range(num_of_batches_in_epoch)]
        plt.figure()
        plt.title("{} - {} Layer {} Head {}- Conf Along Batches".format(module_type, attn_type, layer, num_head))
        plt.xlabel("Batch")
        plt.ylabel("Confidence")
        plt.plot(batches, confs_of_head)
        plt.show()


##############################################################################


def plot_one_head_conf_as_function_of_epoch(directory, experiment, num_of_epochs, module_type, attn_type, head_layer,
                                            head_num):
    confs = []
    epochs = []
    for epoch in range(1, num_of_epochs + 1):
        epochs.append(epoch)
        filename = directory + "/" + experiment + "-epoch-{}".format(epoch)
        print("File name : {}".format(filename))
        with open(filename, 'rb') as fd:
            epoch_data = pickle.load(fd)
        num_of_batches_in_epoch = (epoch_data[module_type][head_layer][attn_type].shape[0])
        confs_tmp = []
        for batch in range(num_of_batches_in_epoch):
            epoch_data_tmp = epoch_data[module_type][head_layer][attn_type][batch]
            # epoch_data_denormalized = denorm_conf(epoch_data_tmp)
            confs_tmp.append(epoch_data_tmp[head_num])
        confs.append(sum(confs_tmp) / len(confs_tmp))
    print(confs)
    plt.title("{} - {} Head {} Layer {} - Conf Along Training".format(module_type, attn_type, head_num, head_layer))
    plt.xlabel("Epoch")
    plt.ylabel("Confidence")
    plt.plot(epochs, confs)


##############################################################################

def plot_heat_map_of_specific_epoch(number_of_layers, number_of_heads, epoch_file, module_type, attn_type):
    layers = [layer for layer in range(number_of_layers)]
    heads = [head for head in range(number_of_heads)]
    with open(epoch_file, 'rb') as fd:
        epoch_data = pickle.load(fd)
    epoch_grid = np.zeros(shape=(number_of_layers, number_of_heads))
    num_of_batches_in_epoch = (epoch_data[module_type][0][attn_type].shape[0])
    for l in layers:
        for h in heads:
            confs_tmp = []
            for batch in range(num_of_batches_in_epoch):
                confs_tmp.append(epoch_data[module_type][l][attn_type][batch][h])
            epoch_grid[l][h] = round(sum(confs_tmp) / len(confs_tmp), 3)

    ax = plt.axes()
    sns.heatmap(epoch_grid, ax=ax, cmap="Blues")
    ax.set_xlabel("heads")
    ax.set_ylabel("layers")
    ax.set_title('{} - {}'.format(module_type, attn_type))
    plt.show()


def plot_heat_map_of_all_epochs(num_of_epochs, epochs_dir, number_of_layers, number_of_heads, module_type, attn_type,
                                experiment, model_phase):
    layers = [layer for layer in range(number_of_layers)]
    heads = [head for head in range(number_of_heads)]
    epoch_grid = np.zeros(shape=(number_of_layers, number_of_heads))
    for epoch in range(1, num_of_epochs + 1):
        filename = epochs_dir + "/" + experiment + "-epoch-{}".format(epoch)
        with open(filename, 'rb') as fd:
            epoch_data = pickle.load(fd)
            num_of_batches_in_epoch = (epoch_data[module_type][0][attn_type].shape[0])
        for l in layers:
            for h in heads:
                confs_tmp = []
                for batch in range(num_of_batches_in_epoch):
                    confs_tmp.append(epoch_data[module_type][l][attn_type][batch][h])
                epoch_grid[l][h] += (sum(confs_tmp) / len(confs_tmp)) / num_of_epochs

    ax = plt.axes()
    sns.heatmap(epoch_grid, ax=ax, cmap="Blues")
    ax.set_xlabel("heads")
    ax.set_ylabel("layers")
    title = '{} : {} - {}'.format(model_phase, module_type, attn_type)
    ax.set_title(title)
    plt.savefig("output_dir/{}.png".format(title))


def plot_alphas(name, epoch_data, attn_tyep, module_type):
    '''

    :param name: Experiment name
    :param epoch_data: A loaded pickle contained the alphas fata.
    :param attn_tyep: Type of attention.
    :param module_type: decoder\encoder
    :return: plot the figure
    '''
    cdict1 = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.1),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
              }
    fig, axs = plt.subplots(2, 3, figsize=(16, 19))
    for ax in axs.flat:
        ax.set(xlabel='heads', ylabel='heads')
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    fig.suptitle('{} - {} - {} '.format(name, module_type, attn_tyep), fontsize=16)
    im1 = axs[0, 0].imshow(epoch_data[module_type][0][attn_tyep], interpolation='nearest', cmap=blue_red1)
    axs[0, 0].set_title("Layer 0")
    fig.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].imshow(epoch_data[module_type][1][attn_tyep], interpolation='nearest', cmap=blue_red1)
    fig.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title("Layer 1")
    im3 = axs[0, 2].imshow(epoch_data[module_type][2][attn_tyep], interpolation='nearest', cmap=blue_red1)
    fig.colorbar(im3, ax=axs[0, 2])
    axs[0, 2].set_title("Layer 2")
    im4 = axs[1, 0].imshow(epoch_data[module_type][3][attn_tyep], interpolation='nearest', cmap=blue_red1)
    fig.colorbar(im4, ax=axs[1, 0])
    axs[1, 0].set_title("Layer 3")
    im5 = axs[1, 1].imshow(epoch_data[module_type][4][attn_tyep], interpolation='nearest', cmap=blue_red1)
    fig.colorbar(im5, ax=axs[1, 1])
    axs[1, 1].set_title("Layer 4")
    im6 = axs[1, 2].imshow(epoch_data[module_type][5][attn_tyep], interpolation='nearest', cmap=blue_red1)
    fig.colorbar(im6, ax=axs[1, 2])
    axs[1, 2].set_title("Layer 5")
    plt.show()


epoch_dir = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs/exp-enc_dec-attn-swaps-layers_04_15-8-heads-6l_with_conf_work_monkey_gamma_20_enc_g_mm_dec_e_g_start_late_40p_nd_decay"
plot_heat_map_of_all_epochs(36, epoch_dir, 6, 8, "encoder", "self_attn",
                            "exp-enc_dec-attn-swaps-layers_04_15-8-heads-6l_with_conf_work_monkey_gamma_20_enc_g_mm_dec_e_g_start_late_40p_nd_decay",
                            "training")
output_dir = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/Plots"
