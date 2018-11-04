import torch
import numpy as np
from torch.utils.data import Dataset
from lib.utils.config import cfg
from lib.utils import provider

class ModelNet40(Dataset):
    def __init__(self, root, train=True):
        filelist = provider.get_file_list(cfg.TRAIN_LIST if train else cfg.TEST_LIST)
        data = []
        label = []
        for f in filelist:
            d, l = provider.get_data(f)
            data.append(d)
            label.append(l)
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        self.len = self.data.shape[0]
        self.num_points = self.data.shape[1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index])
        label = torch.from_numpy(self.label[index])
        
        return data, label
    
"""
test
"""

dataset = ModelNet40(cfg.MODELNET, train=False)
print(len(dataset))
data, label = dataset[10]
print(data.size(), label.size())
