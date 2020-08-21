import pickle
import os
directory = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs"
for filename in os.listdir(directory):
    print(os.path.join(directory, filename))