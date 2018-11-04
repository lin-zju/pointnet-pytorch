from easydict import EasyDict
import os
import sys

cfg = EasyDict()

"""
Path setting
"""


cfg.UTILS_DIR = os.path.dirname(os.path.realpath(__file__))
cfg.LIB_DIR = os.path.dirname(cfg.UTILS_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')

def add_path():
    for key in cfg:
        if 'DIR' in key:
            sys.path.append(cfg[key])

add_path()
"""
Dataset settings
"""

cfg.MODELNET = os.path.join(cfg.DATA_DIR, 'modelnet40_ply_hdf5_2048')
cfg.TRAIN_LIST = os.path.join(cfg.MODELNET, 'train_files.txt')
cfg.TEST_LIST = os.path.join(cfg.MODELNET, 'test_files.txt')
