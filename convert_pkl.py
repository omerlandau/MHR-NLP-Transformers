import pickle
import os
directory = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs"
for filename in os.listdir(directory):
    with open(filename, 'wb') as fd:
        tmp = pickle.load(fd)
        pickle.dump(tmp, fd, protocol=3)
