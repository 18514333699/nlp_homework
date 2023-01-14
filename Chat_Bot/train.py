from dataset import train_data_loader
from seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
import config
from tqdm import tqdm
import torch

'''
训练流程:
1、实例化model,optimizer,loss
2、遍历dataloader
3、调用的output
4、计算损失
5、模型保存和加载
'''
seq2seq = Seq2Seq()
seq2seq = seq2seq.to(config.device)
optimizer = Adam(seq2seq.parameters(), lr=0.001)


def Train(epoch):

    bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ascii=True, desc="train")
    for idx, (input, target, input_length, target_length) in bar:

        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_length = target_length.to(config.device)

        optimizer.zero_grad()

        decoder_outputs,_ = seq2seq(input, target, input_length, target_length)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1), -1)
        target = target.view(-1)
        loss = F.nll_loss(decoder_outputs, target, ignore_index=config.ws_target.PAD)
        loss.backward()
        optimizer.step()
        bar.set_description("epoch:{}\tidx:{}\t loss:{:.3f}".format(epoch, idx, loss.item()))
        if idx % 100 == 0:
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)


#if __name__ == '__main__':

