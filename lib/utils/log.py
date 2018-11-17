from tensorboardX import SummaryWriter
from lib.utils.config import cfg
import os


class Recorder:
    """
    Logging utility
    """
    
    def __init__(self, print_every):
        """
        Contructor
        :param print_every: show results every steps
        """
        self.print_every = print_every
        log_path = os.path.join(cfg.ROOT_DIR, cfg.LOG_DIR)
        self.writer = SummaryWriter(log_dir=log_path)
        self.count = 0
        self.meter = AverageMeter()
    
    def update(self, mode, epoch, step=None, data=None, output=None, step_num=None, loss=None, eval=None):
        """
        Update logging information
        :param mode: 'train', 'val' or 'test'
        :param epoch: training epoch
        :param step: training step
        :param data: data from Dataset
        :param output: output from Network
        :param loss: loss from Criterion
        :param eval: evaluation result from Evaluator
        :return: None
        """
        if mode == 'train':
            self.update_train_loss(epoch, step, loss, step_num)
        elif mode == 'val':
            self.update_val_acc(epoch, eval)
    
    def update_train_loss(self, epoch, step, loss, step_num):
        self.meter.update(loss)
        if step % self.print_every == 0:
            self.writer.add_scalar('loss', self.meter.avg, epoch * step_num + step)
            print('epoch {} step {} loss: {:.2f}'.format(epoch, step, self.meter.avg))
            self.meter.reset()
    
    def update_val_acc(self, epoch, acc):
        self.writer.add_scalar('val_acc', acc, epoch)
    
    def update_learning_rate(self, epoch, lr):
        self.writer.add_scalar('learning_rate', lr, epoch)


class AverageMeter():
    """
    Store and compute statistics from a sequence of values
    Records:
        val: last updated value
        sum: sum of the sequence
        avg: average
        count: count
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val):
        """
        Update the meter
        :param val: updated value
        """
        
        self.count += 1
        self.val = val
        self.sum += val
        self.avg = self.sum / self.count
