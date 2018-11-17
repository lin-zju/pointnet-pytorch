from easydict import EasyDict
import os
import sys
import torch

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
cfg.NUM_CLASS = 40

NAME_TO_ID = {
    'airplane': 0,
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'bookshelf': 4,
    'bottle': 5,
    'bowl': 6,
    'car': 7,
    'chair': 8,
    'cone': 9,
    'cup': 10,
    'curtain': 11,
    'desk': 12,
    'door': 13,
    'dresser': 14,
    'flower_pot': 15,
    'glass_box': 16,
    'guitar': 17,
    'keyboard': 18,
    'lamp': 19,
    'laptop': 20,
    'mantel': 21,
    'monitor': 22,
    'night_stand': 23,
    'person': 24,
    'piano': 25,
    'plant': 26,
    'radio': 27,
    'range_hood': 28,
    'sink': 29,
    'sofa': 30,
    'stairs': 31,
    'stool': 32,
    'table': 33,
    'tent': 34,
    'toilet': 35,
    'tv_stand': 36,
    'vase': 37,
    'wardrobe': 38,
    'xbox': 39,
}

ID_TO_NAME = {}

for item in NAME_TO_ID.items():
    ID_TO_NAME[item[1]] = item[0]

"""
Logging settings
"""

cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'runs/log')
cfg.PRINT_EVERY = 20

"""
Training settings
"""
cfg.NUM_EPOCHS = 100
cfg.BATCH_SIZE = 32
cfg.DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
cfg.PATIENCE = 5
cfg.RESUME = True

"""
Model saving
"""
cfg.CHECKPOINT_DIR = os.path.join(cfg.ROOT_DIR, 'checkpoint')
cfg.BEST_MODEL_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')
cfg.BEST_ACC_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'best_acc.txt')
cfg.CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'exp-1.tar')
