import os
def plot_one_head_conf_as_function_of_epoch(experiment,num_of_epochs,head_layer,head_num):
  confs = []
  epochs = []
  directory = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs/{}".format(experiment)
  for epoch in range(num_of_epochs):
      epochs.append(epoch)
      print(directory)
      for filename in sorted(os.listdir(directory)):
        print(filename)
plot_one_head_conf_as_function_of_epoch("exp-enc_dec-attn-swaps-layers_04_15-8-heads-6l_with_conf_work_monkey_gamma_20_enc_g_mm_dec_e_g_start_late_40p_nd_decay",36,0,0)