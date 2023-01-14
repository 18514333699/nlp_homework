import pickle
import torch

by_word = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



input_path_by_word = r"D:\python-workplace\Chat_Bot1\input_by_word.txt"
target_path_by_word = r"D:\python-workplace\Chat_Bot1\target_by_word.txt"

input_path = r"D:\python-workplace\Chat_Bot1\input.txt"
target_path = r"D:\python-workplace\Chat_Bot1\target.txt"


if by_word:
    ws_input_path = "model/ws_input_path.pkl"
    ws_target_path = "model/ws_target_path.pkl"
else:
    ws_input_path = "model/ws_input_path.pkl"
    ws_target_path = "model/ws_target_path.pkl"

ws_input = pickle.load(open(ws_input_path,"rb"))
ws_target = pickle.load(open(ws_target_path,"rb"))


chatbot_batch_size = 128

if by_word:
    input_max_len = 25
    target_max_len = 25
else:
    input_max_len = 15
    target_max_len = 15

embedding_dim = 256
encoder_num_layers = 1
encoder_hidden_size = 128

decoder_num_layers = 1
decoder_hidden_size = 128

teacher_forcing_ratio = 0.3

model_save_path = "model/seq2seq.model" if by_word else "model/seq2seq_by_word.model"
optimizer_save_path = "model/optimizer.model" if by_word else "model/optimizer_by_word.model"
