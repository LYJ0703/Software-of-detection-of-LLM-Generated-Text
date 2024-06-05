import  six, json
import regeneration
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
import argparse
from utils import tokenize
import numpy as np
import time


r = 0.7
num = 15

min_len = int(12/r)
PorterStemmer1 = PorterStemmer()

def spilt_by_r(answers):
    truncated_answers = []
    rest_answers = []
    for answer in answers:
        # print(answer)
        """
        if len(answer) < min_len:
            print("可能存在无效切割")
            exit()
        """
        gap = int(0.7*(len(answer) - 1))
        while gap < len(answer) and answer[gap] != ' ' :
            # print(answer[gap])
            gap += 1
        truncated_answers.append(answer[:gap])
        rest_answers.append(answer[gap+1:])
    return truncated_answers, rest_answers


def split_string_by_ratio(input_string):

    # 计算分割位置
    ratio=0.7
    split_index = int(len(input_string) * ratio)
    # 分割字符串
    first_part = input_string[:split_index]
    second_part = input_string[split_index:]
    # 返回分割后的两部分字符串
    return first_part, second_part
def get_score_ngrams(target_ngrams, prediction_ngrams):
    # 初始化交集 n-grams 计数为 0
    intersection_ngrams_count = 0
    # 创建一个空字典，用于存储每个 n-gram 的数量
    ngram_dict = {}
    # 遍历目标 n-grams 中的每个 n-gram
    for ngram in six.iterkeys(target_ngrams):
        # 计算交集 n-grams 的数量，取目标 n-grams 和预测 n-grams 中对应 n-gram 数量的最小值
        intersection_ngrams_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
        # 将每个 n-gram 的数量添加到 ngram_dict 中
        ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
    # 计算目标 n-grams 的总数量
    target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
    # 返回交集 n-grams 数量除以目标 n-grams 总数量的比例，以及 ngram_dict 字典
    return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict
def get_ngram_info(article_tokens, summary_tokens, _ngram):
    # 创建文章和摘要的 n-grams
    article_ngram = _create_ngrams(article_tokens, _ngram)
    summary_ngram = _create_ngrams(summary_tokens, _ngram)
    # 调用 get_score_ngrams 函数计算文章和摘要之间的 n-gram 分数
    ngram_score, ngram_dict = get_score_ngrams(article_ngram, summary_ngram)
    # 计算 n-gram 字典中的 n-gram 数量之和
    ngram_count = sum(ngram_dict.values())
    # 返回 n-gram 分数、n-gram 字典和 n-gram 数量之和
    return ngram_score, ngram_dict, ngram_count
def N_gram_detector(ngram_n_ratio):
    score = 0
    non_zero = []

    # 遍历 n-gram_n_ratio
    for idx, key in enumerate(ngram_n_ratio):
        # 如果 idx 在范围内并且 key 包含 'score' 或 'ratio'
        if idx in range(3):
            # 计算得分
            score += 0.0  # 直接加零，无需乘以 ngram_n_ratio[key]
        # 如果 key 包含 'score' 或 'ratio'
        else:
            # 将 ngram_n_ratio[key] 转换为浮点数
            ngram_value = float(ngram_n_ratio[key])
            # 计算得分
            score += (idx + 1) * np.log((idx + 1)) * ngram_value
            # 如果得分不为零，则将 idx+1 添加到非零列表中
            if ngram_value != 0:
                non_zero.append(idx + 1)

    # 返回得分
    return score / (sum(non_zero) + 1e-8)


def get_textlist_Bscore(truncated_answers, rest_answers):
    for truncated_answer, rest_answer in zip(truncated_answers, rest_answers):
        tokens = tokenize(rest_answer, stemmer=PorterStemmer1)
        LLM_answers_list = regeneration.Regeneration(truncated_answer)

        temp = []
        BScore = 0.0
        for regeneration_answer in LLM_answers_list: # len(human_half)
            temp2 = {}
            LLM_answer_tokens=tokenize(regeneration_answer, stemmer=PorterStemmer1)
            for _ngram in range(1, 25):
                ngram_score, ngram_dict, overlap_count = get_ngram_info(tokens,LLM_answer_tokens ,_ngram)
                #temp2['gpt_truncate_ngram_{}_score'.format(_ngram)] = ngram_score / len(LLM_answer_tokens)
                temp2['gpt_truncate_ngram_{}_score'.format(_ngram)] = ngram_score / max(len(LLM_answer_tokens), 1)
            BScore  = BScore  + N_gram_detector(temp2)

                #temp2['gpt_truncate_ngram_{}_count'.format(_ngram)] = overlap_count
            temp.append(temp2)
        BScore =BScore /num
    return BScore




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='An argparse example')
    parser.add_argument('--text', '-t', default='hello', type=str, help="text")
    args = parser.parse_args()

    answers = []
    answers.append(args.text)
    truncated, rest = spilt_by_r(answers)

    BScore = get_textlist_Bscore(truncated, rest)
    print("Bscore:")
    print(BScore)
    with open('Bscore.txt', 'w') as f:
        f.write(str(BScore))


