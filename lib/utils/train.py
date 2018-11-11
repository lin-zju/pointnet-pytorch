import torch
from lib.utils.config import cfg
from lib.networks.pointnet import PointNet
from torch import optim
from lib.utils.loss import PointNetLoss
from torch.utils.data import DataLoader
from lib.datasets.modelnet40 import ModelNet40
from lib.utils.log import Recorder


def train(net, optimizer, criterion, dataloader, epoch, device, recorder=None):
    net = net.to(device)
    net.train()
    for i, data in enumerate(dataloader):
        data = [d.to(device) for d in data]
        input, target = data
        pred, trans = net(input)
        loss = criterion(pred, target, trans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if recorder:
            recorder.update_train_loss(loss.item(), epoch, i)


def val(net, dataloader, epoch, device, recorder=None):
    with torch.no_grad():
        net.eval()
        net = net.to(device)
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            input, target = [d.to(device) for d in data]
            pred, _ = net(input)
            correct += (pred.argmax(dim=1) == target).sum().item()
            total += data[0].size(0)
            
        if recorder:
            recorder.update_val_acc(correct / total, epoch)
        return correct / total


def train_net():
    net = PointNet(40)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(net.parameters(), weight_decay=0.001)
    criterion = PointNetLoss(0.001)
    dataset = ModelNet40(cfg.MODELNET)
    val_dataset = ModelNet40(cfg.MODELNET, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    recorder = Recorder(20)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
    epochs = 100
    for epoch in range(epochs):
        train(net, optimizer, criterion, dataloader, epoch, device, recorder)
        scheduler.step()
        val(net, val_dataloader, epoch, device, recorder)


if __name__ == '__main__':
    train_net()
