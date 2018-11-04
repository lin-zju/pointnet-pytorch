import h5py
import numpy as np
from config import cfg
import os

def get_file_list(filename):
    """
    get a list of file names listed in a file
    """
    return [os.path.join(cfg.ROOT_DIR, line.strip())for line in open(filename, 'r')]

def get_data(filename):
    """
    get data stored in a .h5 file
    """
    f = h5py.File(filename, 'r')
    # slicing produces numpy array
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

"""
test
"""

if __name__ == '__main__':
    import os
    data = []
    label = []
    for f in get_file_list(cfg.TRAIN_LIST):
        f = os.path.join(cfg.ROOT_DIR, f)
        d, l = get_data(f)
        data.append(d)
        label.append(d)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)
    print(data.shape, label.shape)
    print(data.max(), data.min())
    
