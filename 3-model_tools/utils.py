import argparse
import logging
import os
import random
import time
import warnings
import numpy as np
import torch


def set_logging(args):
    '''
    :param args:
    :return:
    '''
    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


def getargs():
    '''
    :param from bash file or terminal line
    :return: args
    '''
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--input_dir', default="./data/train.csv")
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--save_dir', default='./log', help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', default="facebook/bart-base")
    parser.add_argument('--ckpt', default="save.ckpt", help="the checkpoint name")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="gpu or cpu")
    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--valid_batch_size', default=16, type=int)
    parser.add_argument('--valid_epoch', default=1, type=int, help="run validatin ")
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_train_epochs', default=25, type=int)
    parser.add_argument('--save_steps', default=500, type=int)
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    args = parser.parse_args()
    seed_everything(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
