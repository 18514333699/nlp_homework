'''
编码器
'''
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import config


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws_input),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.ws_input.PAD
                                      )
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          num_layers=config.encoder_num_layers,
                          hidden_size=config.encoder_hidden_size,
                          batch_first=True
                          )

    def forward(self, input, input_length):
        '''

        :param input:
        :param input_length:
        :return:
        '''
        embedded = self.embedding(input)
        embedded = pack_padded_sequence(embedded, input_length, batch_first=True)  # 打包
        out,hidden = self.gru(embedded)
        out,out_length = pad_packed_sequence(out, batch_first=True, padding_value=config.ws_input.PAD)  # 解包
        return out, hidden
