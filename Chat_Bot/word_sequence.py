

class WordSequence:
    PAD_TAG = "PAD  "  # 填充词
    UNK_TAG = "UNK  "  # 未知词
    SOS_TAG = "SOS  "  # 开始符
    EOS_TAG = "EOS  "  # 结束符
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS,
                     }
        self.count = {}

    def fit(self, sentence):
        '''
           传入句子，词频统计
           :param sentence:
           :return:
           '''
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=5, max_count=None, max_features=None):

        '''
        构造字典
        :param self:
        :return:
        '''
        temp = self.count.copy()
        for key in temp:
            cur_count = self.count.get(key, 0)
            if min_count is not None:
                if cur_count < min_count:
                    del self.count[key]
            if max_count is not None:
                if cur_count > max_count:
                    del self.count[key]
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_features])
        for key in self.count:
            self.dict[key] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len, add_eos=False):
        '''
        把sentence转化为数字序列
        :param sentence:
        :param max_len:
        :return:
        add_eos:true:输出句子的长度为max_len+1
        add_eos:False:输出句子的长度为max_len
        '''

        if len(sentence) > max_len:
            sentence = sentence[:max_len]  # 保留前部分

        sentence_len = len(sentence)  # 提前计算句子的长度，实现odd_eos后，句子长度统一
        if add_eos:
            sentence = sentence + [self.EOS_TAG]

        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)  # 进行填充

        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        # 把序列转回字符串
        #return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]
        result = []
        for i in indices:
            if i == self.EOS:
                break
            result.append(self.inverse_dict.get(i, self.UNK_TAG))
        return result

    def __len__(self):
        return len(self.dict)

# if __name__ =='__main__':
# num_sequence = Num_Sequence()
# print(num_sequence.dict)
