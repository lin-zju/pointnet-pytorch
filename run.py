from lib.utils.train import train_net, test_net
from lib.utils.args import args


if __name__ == '__main__':
    if args.test:
        test_net()
    else:
        train_net()
