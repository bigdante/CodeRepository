import torch
import numpy as np
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset, SubsetRandomSampler


class MyDataset(Dataset):
    def __init__(self, path):
        inputs, targets = [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.replace("d(", "").replace(")/d", "")
                line_split = line.strip().split('=')
                data = line_split[0]
                target = line_split[1]
                inputs.append(data)
                targets.append(target)

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        input_ids = tokenizer.batch_encode_plus(inputs, max_length=28, pad_to_max_length=True,
                                                truncation=True)
        self.source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
        self.source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
        target_ids = tokenizer.batch_encode_plus(targets, max_length=28, pad_to_max_length=True,
                                                 truncation=True)
        self.target_ids = np.array(target_ids['input_ids'], dtype=np.int32)

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        target_ids = torch.LongTensor(self.target_ids[index])
        return source_ids, source_mask, target_ids

    def __len__(self):
        return len(self.source_ids)


def set_trainloader_validloader(dataset, args, train_split=0.7, shuffle_dataset=True):
    '''

    :param dataset:
    :param args:
    :param train_split:
    :param shuffle_dataset:
    :return:
    '''
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    # shuffle to get different trainset
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size,
                                               sampler=train_sampler, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.valid_batch_size,
                                                    sampler=valid_sampler, shuffle=False)
    return train_loader, validation_loader


def set_trainset_valset(args, train_split=0.7):
    dataset = args.dataset
    length = len(dataset)
    train_size = int(train_split * length)
    validate_size = length - train_size
    # first param is data set to be saperated, the second is list stating how many sets we want it to be.
    train_set, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False)
    return train_set, validate_set, train_loader, validation_loader
