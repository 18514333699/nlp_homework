'''
解码器
'''

import torch.nn as nn
import torch.nn.functional as F
import config
import random
import torch


# from attention import Attention


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws_target),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.ws_target.PAD
                                      )
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.decoder_hidden_size,
                          num_layers=config.decoder_num_layers,
                          batch_first=True
                          )
        self.fc = nn.Linear(config.decoder_hidden_size, len(config.ws_target))
        # self.attn = Attention()
        # self.Wa = nn.linear(config.encoder_hidden_size + config.decoder_hidden_size, config.decoder_hidden_size,bia=False)

    def forward(self, target, encoder_hidden):
        # 获取encoder的输出，作为decoder的第一次隐藏层
        decoder_hidden = encoder_hidden
        batch_size = target.size(0)
        # 准备decoder第一个输入
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * config.ws_target.SOS).to(
            config.device)
        # 保存预测结果
        decoder_outputs = torch.zeros([batch_size, config.target_max_len + 1, len(config.ws_target)]).to(config.device)
        if random.random() > config.teacher_forcing_ratio:  # 使用teacher forcing机制，加速收敛
            for t in range(config.target_max_len + 1):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output_t
                decoder_input = target[:, t].unsqueeze(-1)
        else:  # 不使用teacher forcing机制
            for t in range(config.target_max_len + 1):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output_t
                value, index = torch.topk(decoder_output_t, 1)
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        '''

        :param decoder_input:
        :param decoder_hidden:
        :return:
        '''
        decoder_input_embedded = self.embedding(decoder_input)
        out, decoder_hidden = self.gru(decoder_input_embedded, decoder_hidden)
        out = out.squeeze(1)
        ouput = F.log_softmax(self.fc(out), dim=-1)
        return ouput, decoder_hidden

    def evaluate(self, encoder_hidden):
        '''
        评估
        :param encoder_hidden:
        :return:
        '''
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * config.ws_target.SOS).to(
            config.device)

        indices = []
        for i in range(config.target_max_len + 5):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index
            indices.append(index.squeeze(-1).cpu().detach().numpy())
        return indices
