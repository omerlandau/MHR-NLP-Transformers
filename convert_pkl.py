import pickle
import os
directory = "/specific/netapp5_2/gamir/edocohen/guy_and_brian/guy/omer_temp/MHR-runs/confs"
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    print(file)
    with open(file, 'rb') as fd:
        tmp = pickle.load(fd)
    fd.close()
    with open(os.path.join(directory, filename), 'wb') as fd2:
        pickle.dump(tmp, fd, protocol=3)
    fd.close()
