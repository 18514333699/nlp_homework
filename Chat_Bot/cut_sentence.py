# import config
import string
import jieba.posseg as psg

# from lib import stopwords

# jieba.load_userdict(config.user_dict_path)

letters = string.ascii_lowercase  # 准备英文字符


def cut_sentence_by_word(sentence):
    '''实现中英文分词'''
    # python和c++哪个难
    temp = ""
    result = []
    for word in sentence:
        # 把英文单词进行拼接
        if word.lower() in letters:
            temp += word
        else:
            if temp != "":  # 出现中文，把英文加到结果中
                result.append(temp.lower())
                temp = ""
            result.append(word.strip())
    if temp != "":
        result.append(temp.lower())
    return result


def cut(sentence, by_word=False, use_stopwords=False, with_sg=False):
    '''

    :param sentence: str 句子
    :param by_word: 是否按照单个词进行分词
    :param use_word: 是否返回词性
    :param with_sg:
    :return:
    '''
    if by_word:
        result = cut_sentence_by_word(sentence)
    else:
        result = psg.lcut(sentence)
        result = [(i.word, i.flag) for i in result]

        if not with_sg:
            result = [i[0] for i in result]
            '''
    # 是否使用停用词
    if use_stopwords:
        result = [i for i in result if i not in stopwords]
        '''
    return result
