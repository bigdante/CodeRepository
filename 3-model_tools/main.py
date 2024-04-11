from data_utils.data import *
from models.model import *
from train import *


def train(args):
    set_logging(args)
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    # get dataset and loader
    args.dataset = MyDataset(args.input_dir)
    args.train_set, args.valid_set, args.train_loader, args.validation_loader = set_trainset_valset(args, 0.8)
    # set model
    model = BasicModel(args)
    model.train_model()



if __name__ == '__main__':
    args = getargs()
    train(args)
    test(args)
