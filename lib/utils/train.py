import torch
from lib.config.config import cfg
from lib.config.config import ID_TO_NAME
from lib.networks.pointnet import PointNet
from torch import optim
from lib.utils.loss import PointNetLoss
from torch.utils.data import DataLoader
from lib.datasets.modelnet40 import ModelNet40
from lib.utils.log import Recorder
from lib.utils.checkpoint import save_best_model, load_best_model
from lib.utils.checkpoint import save_checkpoint, load_checkpoint


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
            recorder.update(mode='train', epoch=epoch, step=i, step_num=len(dataloader), loss=loss.item())


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
            recorder.update(mode='val', epoch=epoch, eval=correct / total)
        return correct / total


def schedule_learning_rate(scheduler, optimizer, epoch, eval_result, recorder):
    scheduler.step(eval_result)
    learning_rate = optimizer.param_groups[0]['lr']
    recorder.update_learning_rate(epoch, learning_rate)


def test(net, dataloader, device):
    with torch.no_grad():
        net.eval()
        net = net.to(device)
        total = {label: 0 for label in range(cfg.NUM_CLASS)}
        correct = {label: 0 for label in range(cfg.NUM_CLASS)}
        for i, data in enumerate(dataloader):
            input, target = [d.to(device) for d in data]
            scores, _ = net(input)
            for i, score in enumerate(scores):
                truth = target[i].item()
                pred = score.argmax().item()
                correct[truth] += (truth == pred)
                total[truth] += 1
        accuracy = {label: correct[label] / total[label] for label in total}
        
        print('\nEvaluation Results:')
        
        for label in accuracy:
            print('{}: {:.2f}'.format(ID_TO_NAME[label], accuracy[label]))
        
        print('\naverage: {:.2f}'.format(sum(accuracy.values()) / len(accuracy)))


def train_net():
    device = cfg.DEVICE
    net = PointNet(40).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = PointNetLoss(0.001)
    dataset = ModelNet40(cfg.MODELNET)
    val_dataset = ModelNet40(cfg.MODELNET, train=False)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    recorder = Recorder(cfg.PRINT_EVERY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=cfg.PATIENCE)
    
    # resume lsat training
    start_epoch = 0
    if cfg.RESUME:
        start_epoch = load_checkpoint(net, optimizer, scheduler)
    
    # start training
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        train(net, optimizer, criterion, dataloader, epoch, device, recorder)
        acc = val(net, val_dataloader, epoch, device, recorder)
        schedule_learning_rate(scheduler, optimizer, epoch, acc, recorder)
        save_checkpoint(net, optimizer, scheduler, epoch)
        save_best_model(net, acc)
    test(net, val_dataloader, device)


def test_net():
    device = cfg.DEVICE
    dataset = ModelNet40(cfg.MODELNET, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    net = PointNet(40).to(device)
    load_best_model(net)
    test(net, dataloader, device)


if __name__ == '__main__':
    test_net()
    # train_net()
