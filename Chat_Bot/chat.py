from cut_sentence import cut
import torch
import config
from seq2seq import Seq2Seq
import numpy as np


# 模拟聊天,对用户输入进来的话进行回答
def chat(by_word=True):
    seq2seq = Seq2Seq()
    seq2seq = seq2seq.to(config.device)
    seq2seq.load_state_dict(torch.load(config.model_save_path))

    while True:
        input_string = input('I：')
        input_string = cut(input_string,by_word=by_word)
        input_length = torch.LongTensor([len(input_string) if len(input_string)>config.input_max_len else config.input_max_len]).to(config.device)
        input_string = torch.LongTensor([config.ws_input.transform(input_string,max_len=config.input_max_len)]).to(config.device)
        indices = np.array(seq2seq.evaluate(input_string,input_length)).flatten()
        outputs = "".join(config.ws_target.inverse_transform(indices))
        print("you：",outputs)


if __name__ == '__main__':
    chat()