import torch
import os
from lib.utils.config import cfg


def save_best_model(model, acc):
    if not os.path.exists(cfg.BEST_MODEL_DIR):
        torch.save(model.state_dict(), cfg.BEST_MODEL_DIR)
        open(cfg.BEST_ACC_DIR, 'w').write(str(acc))
    else:
        best_acc = float(open('best_acc.txt', 'r').read())
        if acc > best_acc:
            torch.save(model.state_dict(), cfg.BEST_MODEL_DIR)
            open(cfg.BEST_ACC_DIR, 'w').write(str(acc))


def save_checkpoint(model, optimizer, scheduler, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, cfg.CHECKPOINT_DIR)


def load_checkpoint(model, optimizer, scheduler):
    """
    Load checkpoint into model and optimizer, returns epoch left
    :return:
        the epoch to start from
    """
    if not os.path.exists(cfg.CHECKPOINT_DIR):
        print('checkpoint {} does not exists.'.format(cfg.CHECKPOINT_DIR))
    checkpoint = torch.load(cfg.CHECKPOINT_DIR)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'] + 1
