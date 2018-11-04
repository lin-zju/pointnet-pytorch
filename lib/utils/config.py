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


"""
Dataset settings
"""

cfg.MODELNET_DIR = os.path.join(cfg.DATA_DIR, 'modelnet40_ply_hdf5_2048')
cfg.TRAIN_LIST = os.path.join(cfg.MODELNET_DIR, 'train_files.txt')
cfg.TEST_LIST = os.path.join(cfg.MODELNET_DIR, 'test_files.txt')
