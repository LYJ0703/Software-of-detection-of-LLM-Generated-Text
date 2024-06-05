import torch
import torch.nn as nn
import json
from collections import Counter
import jsonlines
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from torch.utils.data import DataLoader, TensorDataset
import logging
import  gc
import random
import copy
import torch.optim as optim
import argparse


def evaluate(model, test_data, device, batch_size):
    model.eval()
    predictions = []
    num_batches = (len(test_data) + batch_size - 1) // batch_size

    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(test_data))

        # 获取当前批次的文本和标签
        batch_data = test_data[start_index:end_index]
        # 使用 tokenizer 处理当前批次的文本
        batch_data=torch.Tensor(batch_data).to(device)
        with torch.no_grad():
            output = model(batch_data)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.tolist())
    return predictions


class TextClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(TextClassifier, self).__init__()
        #self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, input_dim *2)
        self.fc2 = nn.Linear(input_dim * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, batch_data):

        model_fc1_output = self.fc1(batch_data)
        activated_output = self.relu(model_fc1_output)
        output = self.fc2(activated_output)
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='bert,xlnet,roberta')
    parser.add_argument('--ppl', type=int, default=0, help='1表示考虑，0表示不考虑')
    parser.add_argument('--bscore', type=int, default=0, help='1表示考虑，0表示不考虑')
    args = parser.parse_args()
    input_dim=0
    if args.model !='':
        input_dim+=1
    if  args.ppl ==1:
        input_dim+=1
        ppl = "ppl"
    else:
        ppl = ""
    if args.bscore==1:
        input_dim+=1
        bscore = "bscore"
    else:
        bscore=""

    Essay_data = []
    Essay_labels = []
    WP_data = []
    WP_labels = []
    Reuters_data = []
    Reuters_labels = []
    if args.model != '':
        Essay_data_softmax = []
        WP_data_softmax = []
        Reuters_data_softmax = []
        with open('../benchmark/' + args.model + '_softmax_Essay.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                Essay_data_softmax.append(data)
        Essay_data_softmax=Essay_data_softmax[699:699+1429]
        with open('../benchmark/' + args.model + '_softmax_WP.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                WP_data_softmax.append(data)
        with open('../benchmark/' + args.model + '_softmax_Reuters.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                Reuters_data_softmax.append(data)
    if args.ppl == 1:
        Essay_data_ppl = []
        WP_data_ppl = []
        Reuters_data_ppl = []
        with open('../benchmark/Essay_LLMs_ppls.json') as f:
            dict=json.load(f)
            ppls=dict['ppl']
            for ppl in ppls:
                Essay_data_ppl.append(ppl)
        Essay_data_ppl=Essay_data_ppl[699:699+1429]
        with open('../benchmark/WP_LLMs_ppls.json') as f:
            dict = json.load(f)
            ppls = dict['ppl']
            for ppl in ppls:
                WP_data_ppl.append(ppl)

        with open('../benchmark/Reuters_LLMs_ppls.json') as f:
            dict = json.load(f)
            ppls = dict['ppl']
            for ppl in ppls:
                Reuters_data_ppl.append(ppl)

    Essay_data_bscore = []
    WP_data_bscore = []
    Reuters_data_bscore = []
    with open('../benchmark/Ess2.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore = list(line.values())[0]
            bscore = bscore * 10000
            Essay_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                Essay_labels.append(0)
            else:
                Essay_labels.append(1)

    with open('../benchmark/WP_output.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore = list(line.values())[0]
            bscore = bscore * 10000
            WP_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                WP_labels.append(0)
            else:
                WP_labels.append(1)

    with open('../benchmark/Reu_output.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore = list(line.values())[0]
            bscore = bscore * 10000
            Reuters_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                Reuters_labels.append(0)
            else:
                Reuters_labels.append(1)
    print(len(Essay_labels))
    for i in range(len(Essay_labels)):
        data = []
        if args.model != '':
            data.append(Essay_data_softmax[i])
        if args.ppl == 1:
            data.append(Essay_data_ppl[i])
        if args.bscore == 1:
            data.append(Essay_data_bscore[i])
        Essay_data.append(data)

    for i in range(len(WP_labels)):
        data = []
        if args.model != '':
            data.append(WP_data_softmax[i])
        if args.ppl == 1:
            data.append(WP_data_ppl[i])
        if args.bscore == 1:
            data.append(WP_data_bscore[i])
        WP_data.append(data)

    for i in range(len(Reuters_labels)):
        data = []
        if args.model != '':
            data.append(Reuters_data_softmax[i])
        if args.ppl == 1:
            data.append(Reuters_data_ppl[i])
        if args.bscore == 1:
            data.append(Reuters_data_bscore[i])
        Reuters_data.append(data)
    device = torch.device("cuda")
    model = TextClassifier(num_classes=2, input_dim=input_dim).to(device)
    if args.ppl == 1:
        ppl = "ppl"
    else:
        ppl = ""
    if args.bscore == 1:
        bscore = "bscore"
    else:
        bscore = ""
    if args.ppl==0 and args.bscore==0:
        file_path='./' + args.model+'.pt'
    else:
        file_path='./' + args.model + '_' + ppl + '_' + bscore + 'classifier.pt'
    model.load_state_dict(
        torch.load(file_path, map_location=device))
    model.eval()
    batch_size=4
    predictions_Essay = evaluate(model, Essay_data, device, batch_size)
    predictions_WP = evaluate(model, WP_data, device, batch_size)
    predictions_Reuters = evaluate(model, Reuters_data, device, batch_size)

    print(args.model)
    print("ppl:", args.ppl)
    print("bscore:", args.bscore)
    print('\n')
    accuracy = accuracy_score(Essay_labels, predictions_Essay)
    precision = precision_score(Essay_labels, predictions_Essay)
    recall = recall_score(Essay_labels, predictions_Essay)
    f1 = f1_score(Essay_labels, predictions_Essay)
    print('\n')
    print("Essay_Accuracy:", accuracy)
    print("Essay_Precision", precision)
    print("Essay_Recall", recall)
    print("Essay_F1", f1)
    print('\n')
    accuracy = accuracy_score(WP_labels, predictions_WP)
    precision = precision_score(WP_labels, predictions_WP)
    recall = recall_score(WP_labels, predictions_WP)
    f1 = f1_score(WP_labels, predictions_WP)
    print('\n')
    print("WP_Accuracy:", accuracy)
    print("WP_Precision", precision)
    print("WP_Recall", recall)
    print("WP_F1", f1)
    print('\n')
    accuracy = accuracy_score(Reuters_labels, predictions_Reuters)
    precision = precision_score(Reuters_labels, predictions_Reuters)
    recall = recall_score(Reuters_labels, predictions_Reuters)
    f1 = f1_score(Reuters_labels, predictions_Reuters)
    print("Reuters_Accuracy:", accuracy)
    print("Reuters_Precision", precision)
    print("Reuters_Recall", recall)
    print("Reuters_F1", f1)
if __name__ == '__main__':
    main()