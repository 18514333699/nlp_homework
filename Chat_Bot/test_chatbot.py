'''
测试
'''
import config
from word_sequence import WordSequence
import pickle
from dataset import train_data_loader
from train import Train


def save_ws():
    ws = WordSequence()
    for line in open(config.input_path_by_word).readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(config.ws_input_path, "wb"))

    ws = WordSequence()
    for line in open(config.target_path_by_word).readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(config.ws_target_path, "wb"))


def test_data_loader():
    for idx, (input, target, input_length, target_length) in enumerate(train_data_loader):
        print(idx)
        print(input)
        print(target)
        print(input_length)
        print(target_length)
        break


def train_seq2seq():
    for i in range(10):
        Train(i)


if __name__ == '__main__':
    # save_ws()
    #test_data_loader()
    train_seq2seq()
