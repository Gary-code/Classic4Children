import torch
import json

from readability import Readability


def map_value_to_score(value, mean=0, variance=1, min_score=0, max_score=100):
    """
    将某个值通过正态分布函数映射为最终的分数。
    
    :param value: 输入值
    :param mean: 正态分布的均值
    :param std_dev: 正态分布的标准差
    :param min_score: 最低分数
    :param max_score: 最高分数
    :return: 映射后的分数
    """
    import scipy.stats as stats
    import numpy as np 
    std_dev = np.sqrt(variance)
    z_score = (value - mean) / std_dev
    
    probability = stats.norm.pdf(z_score)
    
    score = min_score + probability * (max_score - min_score)
    
    return score
def get_readability_score(text):
    from cntext import readability
    def score_gen(text,length,mean_2=0.05,var_2=0.025,mean_4=0.85,var_4=0.425):
        result = readability(text, lang='chinese')
        #print(result)
        readability1=(result['readability1'])
        readability2=(result['readability2'])
        readability3=(result['readability4'])
        readability2_mapped = (map_value_to_score(readability2, mean_2, var_2, 0, 1))
        readability3_mapped = (map_value_to_score(readability3, mean_4, var_4, 0, 1))
        
        score_3 = 1 - readability1 / length
        if score_3 < 0:
            score_3 = 0
        '''
        score_1,连词副词占比
        score_2,字频
        score_3,句子长度
        '''
        score=(0.3*readability2_mapped+0.4*readability3_mapped+0.3*score_3)

        return score
    return score_gen(text,len(text))
    

class MyScorer:
    """
    Scorer using BS-Fact, code adapted from bertscore official repo: https://github.com/Tiiiger/bert_score
    """

    def __init__(self, name_module, readability_score, device="cuda"):
        self.name_module = name_module
        self.readability_score = readability_score
        self.device = device

    def score(self, summaries, index):
        """
        Output the score for each example.
        summaries: The summary strings
        index: The indice of example (document that it should be compared to). IT should ideally be just range() except for beam search.
        """

        readability_scores = []
        for text in summaries:
            try:
                readability_scores.append(get_readability_score(text))
            except:
                readability_scores.append(1)
        readability_scores = [1 - (abs(fs - self.readability_score) / 1) for fs in readability_scores]

        return torch.tensor(readability_scores).to(self.device)

