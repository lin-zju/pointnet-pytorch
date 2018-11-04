import torch
from torch import nn
import torch.nn.functional as F


class TransformLoss(nn.Module):
    def __init__(self):
        super(TransformLoss, self).__init__()
    
    def forward(self, m):
        n = m.size(-1)
        batch_size = m.size(0)
        mT = m.permute(0, 2, 1)
        I = torch.eye(n).to(m.device)
        diff = (mT.bmm(m) - I).view(batch_size, -1)
        loss = torch.norm(diff, dim=1)
        
        return loss.mean()


class PointNetLoss(nn.Module):
    def __init__(self, trans_weight):
        super(PointNetLoss, self).__init__()
        self.trans_weight = trans_weight
        self.trans_loss = TransformLoss()
    
    def forward(self, input, target, trans):
        class_loss = F.cross_entropy(input, target)
        trans_loss = self.trans_loss(trans)
        loss = class_loss + self.trans_weight * trans_loss
        return loss


if __name__ == '__main__':
    trans = torch.zeros(100, 64, 64)
    input = torch.rand(10, 40)
    target = torch.randint(40, (10,)).long()
    criterion = PointNetLoss(0)
    loss = criterion(input, target, trans)
    print(loss)
