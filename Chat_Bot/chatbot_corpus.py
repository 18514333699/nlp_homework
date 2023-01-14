''' 准备语料
 if flag == 0:
                f_input.write(line)
                flag = 1
            else:
                f_target.write(line)
                flag = 0
'''

import string
from tqdm import tqdm
from cut_sentence import cut


def filter(pair):
    if pair[0][1].strip() in list(string.ascii_lowercase):
        return True
    elif pair[1][1].count("=") >= 2 and len(pair[1][0].split()) < 4:
        return True
    elif len(pair[0][0].strip()) == 0 or len(pair[1][0].strip()) == 0:# 过滤空行
        return True


def prepar_chinese_corpus(by_word=False):
    path = r"D:\python-workplace\Chat_Bot1\chinses.txt"
    if by_word:
        input_path = r"D:\python-workplace\Chat_Bot1\input_by_word.txt"
        target_path = r"D:\python-workplace\Chat_Bot1\target_by_word.txt"
    else:
        input_path = r"D:\python-workplace\Chat_Bot1\input.txt"
        target_path = r"D:\python-workplace\Chat_Bot1\target.txt"
    flag = 0
    one_pair = []
    f_input = open(input_path, "a")
    f_target = open(target_path, "a")
    for line in tqdm(open(path, encoding='UTF-8').readlines(), ascii=True, desc="处理语料"):
        if line.startswith("P"):
            continue
        else:
            line = line[1:].strip().lower()
            line_cut = cut(line, by_word=by_word)
            line_cut = " ".join(line_cut)
            if len(one_pair) < 2:
                one_pair.append([line_cut, line])
            if len(one_pair) == 2:
                if filter(one_pair):
                    one_pair = []
                    continue
                f_input.write(one_pair[0][0] + "\n")
                f_target.write(one_pair[1][0] + "\n")
                one_pair = []
    f_input.close()
    f_target.close()
