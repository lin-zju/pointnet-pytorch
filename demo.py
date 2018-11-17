from lib.datasets.modelnet40 import ModelNet40;
from lib.config.config import cfg

dataset = ModelNet40(cfg.MODELNET)
data, label = dataset[1]
print(data.max(), data.min())

