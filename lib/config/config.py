from easydict import EasyDict
from lib.utils.args import args
import os
import sys
import torch
import json


cfg = EasyDict()


"""
Path setting
"""

cfg.CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
cfg.LIB_DIR = os.path.dirname(cfg.CONFIG_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')

def add_path():
    for key in cfg:
        if 'DIR' in key:
            sys.path.append(cfg[key])

add_path()

"""
arg configurations
"""
conf_path = os.path.join(cfg.ROOT_DIR, args.conf)
with open(conf_path, 'r') as f:
    conf = json.load(f)

conf_list = ['run_name', 'resume', 'learning_rate', 'clear_history',
             'weight_decay', 'num_epochs', 'batch_size', 'patience', 'print_every']
for name in conf_list:
    if name not in conf:
        print('configuration file: "{}" missing.'.format(name))
"""
RUN NAME
"""
cfg.RUN_NAME = conf['run_name']


"""
Dataset settings
"""

cfg.MODELNET = os.path.join(cfg.DATA_DIR, 'modelnet40_ply_hdf5_2048')
cfg.TRAIN_LIST = os.path.join(cfg.MODELNET, 'train_files.txt')
cfg.TEST_LIST = os.path.join(cfg.MODELNET, 'test_files.txt')
cfg.NUM_CLASS = 41

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
    'gun': 40
}

ID_TO_NAME = {}

for item in NAME_TO_ID.items():
    ID_TO_NAME[item[1]] = item[0]

"""
Logging settings
"""

cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'runs', cfg.RUN_NAME)
cfg.CLEAR_HISTORY = conf['clear_history']

cfg.PRINT_EVERY = conf['print_every']

"""
Training settings
"""
cfg.NUM_EPOCHS = conf['num_epochs']
cfg.BATCH_SIZE = conf['batch_size']
cfg.DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
cfg.PATIENCE = conf['patience']
cfg.RESUME = conf['resume']
cfg.NUM_WORKERS = conf['num_workers']
cfg.LEARNING_RATE = conf['learning_rate']
cfg.WEIGHT_DECAY = conf['weight_decay']

"""
Model saving
"""
cfg.CHECKPOINT_DIR = os.path.join(cfg.ROOT_DIR, 'checkpoint')
cfg.BEST_MODEL_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')
cfg.BEST_ACC_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'best_acc.txt')
cfg.CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, cfg.RUN_NAME + '.tar')

"""
Clear history
"""

if cfg.CLEAR_HISTORY:
    if os.path.exists(cfg.LOG_DIR):
        for f in os.listdir(cfg.LOG_DIR):
            os.remove(os.path.join(cfg.LOG_DIR, f))
    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
