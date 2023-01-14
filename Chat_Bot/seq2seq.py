'''
encoder与decoder合并成seq2seq模型
'''

import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import config


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)

    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder(target,encoder_hidden)
        return decoder_outputs, decoder_hidden

    def evaluate(self, input, input_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        indices = self.decoder.evaluate(encoder_hidden)
        return indices
