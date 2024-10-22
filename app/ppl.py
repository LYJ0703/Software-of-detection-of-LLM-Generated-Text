#-*- coding:utf-8 -*-
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM,RobertaTokenizer,RobertaForMaskedLM

# 导入你的模型
model = RobertaTokenizer.from_pretrained('../roberta_base')
tokenizer = RobertaTokenizer.from_pretrained('../roberta_base')

# Load pre-trained model (weights)
def ppl_cal(sentence,model,tokenizer):

    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenize_input = tokenizer.tokenize(sentence)
    tokenize_input = tokenize_input[0:128]
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    sen_len = len(tokenize_input)
    sentence_loss = 0.

    for i, word in enumerate(tokenize_input):
        # add mask to i-th character of the sentence
        tokenize_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        output = model(mask_input)

        prediction_scores = output[0]
        softmax = nn.Softmax(dim=0)
        ps = softmax(prediction_scores[0, i]).log()
        word_loss = ps[tensor_input[0, i]]
        sentence_loss += word_loss.item()

        tokenize_input[i] = word

    ppl = np.exp(-sentence_loss/sen_len)
    return ppl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An argparse example')
    parser.add_argument('--text', '-t', default='hello', type=str, help="text")
    args = parser.parse_args()
    
    with open('ppl.txt', 'w') as f:
        score = ppl_cal(args.text, model, tokenizer)
        print(score)
        f.write(str(score))