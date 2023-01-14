'''
数据集的准备
'''

from torch.utils.data import DataLoader, Dataset
import config
import torch
from word_sequence import WordSequence


class CTDataset(Dataset):
    def __init__(self):
        self.input_path = config.input_path_by_word
        self.target_path = config.target_path_by_word
        self.input_lines = open(self.input_path).readlines()
        self.target_lines = open(self.target_path).readlines()
        assert len(self.input_lines) == len(self.target_lines), "input和target长度一致"

    def __getitem__(self, index):
        input = self.input_lines[index].strip().split()
        target = self.target_lines[index].strip().split()
        input_length = len(input) if len(input)<config.input_max_len else config.input_max_len
        target_length = len(target) if len(target)<config.target_max_len else config.target_max_len
        return input, target, input_length, target_length

    def __len__(self):
        return len(self.input_lines)


def collate_fn(batch):
    '''
    把input转化为序列
    :param batch:
    :return:
    '''
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    input, target, input_length, target_length = zip(*batch)

    input = [config.ws_input.transform(i, max_len=config.input_max_len,add_eos=False) for i in input]
    input = torch.LongTensor(input)
    target = [config.ws_target.transform(i, max_len=config.target_max_len, add_eos=True) for i in target]
    target = torch.LongTensor(target)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input, target, input_length, target_length


train_data_loader = DataLoader(CTDataset(), batch_size=config.chatbot_batch_size, shuffle=True,collate_fn=collate_fn)
