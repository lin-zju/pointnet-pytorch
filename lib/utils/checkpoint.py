import torch
import sys
import os
from lib.config.config import cfg


def save_best_model(model, acc):
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
        open(cfg.BEST_ACC_PATH, 'w').write(str(acc))
    else:
        best_acc = float(open('best_acc.txt', 'r').read())
        if acc > best_acc:
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            open(cfg.BEST_ACC_PATH, 'w').write(str(acc))

def load_best_model(model):
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        print('best model does not exists.')
        sys.exit(0)
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH))
        
    


def save_checkpoint(model, optimizer, scheduler, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, cfg.CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler):
    """
    Load checkpoint into model and optimizer, returns epoch left
    :return:
        the epoch to start from
    """
    start_epoch = 0
    if os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    return start_epoch
