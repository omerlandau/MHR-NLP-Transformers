import os
import pickle


def denorm_conf(epoch_data_to_denorm, number_of_heads):
    return [epoch_data_to_denorm[i] * epoch_data_to_denorm[number_of_heads] for i in range(number_of_heads)]


def plot_one_head_conf_as_function_of_epoch(experiment, num_of_epochs, module_type, attn_type, head_layer, head_num):
    confs = []
    epochs = []
    directory = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs/{}".format(experiment)
    for epoch in reversed(range(num_of_epochs)):
        epochs.append(epoch)
        for filename in os.listdir(directory):
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


plot_one_head_conf_as_function_of_epoch(
    "exp-enc_dec-attn-swaps-layers_04_15-8-heads-6l_with_conf_work_monkey_gamma_20_enc_g_mm_dec_e_g_start_late_40p_nd_decay",
    "encoder", "self_attn", 36, 0, 0)
