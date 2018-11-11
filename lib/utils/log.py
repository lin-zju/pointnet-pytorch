from tensorboardX import SummaryWriter
from lib.utils.config import cfg
import os

class Recorder:
    def __init__(self, print_every):
        self.print_every = print_every
        log_path = os.path.join(cfg.ROOT_DIR, 'runs/log')
        self.writer = SummaryWriter(log_dir=log_path)
        self.count = 0
    
    def update_train_loss(self, loss, epoch, step):
        self.count += 1
        self.writer.add_scalar('loss', loss, self.count)
        if step % self.print_every == 0:
            print('epoch {} step {} loss: {:.2f}'.format(epoch, step, loss))
    
    def update_val_acc(self, acc, epoch):
        self.writer.add_scalar('val_acc', acc, epoch)
