import math
import torch
from lib.utils.config import cfg
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, dim_out):
        super(TNet, self).__init__()
        self.dim_out = dim_out
        self.fc1 = nn.Conv1d(dim_out, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc4 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, dim_out * dim_out)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        x = x.view(-1, self.dim_out, self.dim_out)
        return x


class PointNet(nn.Module):
    def __init__(self, cls_num):
        super(PointNet, self).__init__()
        self.trans_input = TNet(3)
        self.fc1 = nn.Conv1d(3, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.trans_feature = TNet(64)
        self.fc3 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc6 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc8 = nn.Linear(256, cls_num)
        
        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
    
    def forward(self, x):
        input_transform = self.trans_input(x)
        x = torch.bmm(input_transform, x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        feature_transform = self.trans_feature(x)
        x = torch.bmm(feature_transform, x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.dropout(x)
        x = self.fc8(x)
        
        return x, feature_transform


"""
Test
"""

if __name__ == '__main__':
    import time
    
    net = PointNet(40)
    print('see')
    time.sleep(10)
    net = net.to(cfg.DEVICE)
    print('see')
    time.sleep(20)
