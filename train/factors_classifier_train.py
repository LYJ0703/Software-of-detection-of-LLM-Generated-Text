import torch
import torch.nn as nn
import json
from collections import Counter
import jsonlines
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from torch.utils.data import DataLoader, TensorDataset
import logging
import gc
import random
import copy
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os
import sys
import math

def train(model, train_data, train_labels, optimizer, criterion, device, batch_size):
    model.train()
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    epoch_losses = []
    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(train_data))

        # 获取当前批次的文本和标签
        batch_data = train_data[start_index:end_index]
        batch_labels = train_labels[start_index:end_index]

        # 使用 tokenizer 处理当前批次的文本
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        output = model(batch_data).to(device)
        loss = criterion(output, batch_labels)
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return epoch_losses


def evaluate(model, test_data, test_labels,device, batch_size):
    model.eval()
    predictions = []
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    epoch_losses = []
    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(test_data))

        # 获取当前批次的文本和标签
        batch_data = test_data[start_index:end_index]
        batch_labels = test_labels[start_index:end_index]
        # 使用 tokenizer 处理当前批次的文本
        batch_data = batch_data.to(device)
        with torch.no_grad():
            output = model(batch_data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, batch_labels)
            epoch_losses.append(loss.item())
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.tolist())
    return predictions,epoch_losses


class TextClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(TextClassifier, self).__init__()
        # self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, batch_data):

        model_fc1_output = self.fc1(batch_data)
        #print(model_fc1_output)
        activated_output = self.relu(model_fc1_output)
        output = self.fc2(activated_output)
        #print(output)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='bert,xlnet,roberta')
    parser.add_argument('--ppl', type=int, default=0, help='1表示考虑，0表示不考虑')
    parser.add_argument('--bscore', type=int, default=0, help='1表示考虑，0表示不考虑')
    args = parser.parse_args()

    num_classes = 2
    epochs = 100
    batch_size = 4

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    eval_data = []
    eval_labels = []

    if args.model != '':
        train_data_softmax = []
        test_data_softmax = []
        eval_data_softmax = []
        with open('../three_data/' + args.model + '_softmax_use.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                train_data_softmax.append(data)
        with open('../three_data/' + args.model + '_softmax_test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                test_data_softmax.append(data)
        with open('../three_data/' + args.model + '_softmax_eval.jsonl') as f:
            for line in jsonlines.Reader(f):
                data = line[1]
                eval_data_softmax.append(data)
    if args.ppl == 1:
        train_data_ppl = []
        test_data_ppl = []
        eval_data_ppl = []
        with open('../three_data/use_ppls.txt') as f:
            for line in f:
                train_data_ppl.append(float(line.strip()))

        with open('../three_data/test_ppls.txt') as f:
            for line in f:
                test_data_ppl.append(float(line.strip()))

        with open('../three_data/eval_ppls.txt') as f:
            for line in f:
                eval_data_ppl.append(float(line.strip()))

    train_data_bscore = []
    test_data_bscore = []
    eval_data_bscore = []
    with open('../three_data/use_bscore.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore=list(line.values())[0]
            bscore=bscore*10000
            train_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                train_labels.append(0)
            else:
                train_labels.append(1)

    with open('../three_data/test_bscore.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore = list(line.values())[0]
            bscore = bscore * 10000
            test_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                test_labels.append(0)
            else:
                test_labels.append(1)

    with open('../three_data/eval_bscore.jsonl') as f:
        for line in jsonlines.Reader(f):
            bscore = list(line.values())[0]
            bscore = bscore * 10000
            eval_data_bscore.append(bscore)
            if 'human_answers' in line.keys():
                eval_labels.append(0)
            else:
                eval_labels.append(1)

    for i in range(len(train_labels)):
        data = []
        if args.model != '':
            data.append(train_data_softmax[i])
        if args.ppl == 1:
            data.append(train_data_ppl[i])
        if args.bscore == 1:
            data.append(train_data_bscore[i])

        train_data.append(data)

    for i in range(len(test_labels)):
        data = []
        if args.model != '':
            data.append(test_data_softmax[i])
        if args.ppl == 1:
            data.append(test_data_ppl[i])
        if args.bscore == 1:
            data.append(test_data_bscore[i])

        test_data.append(data)
    for i in range(len(eval_labels)):
        data = []
        if args.model != '':
            data.append(eval_data_softmax[i])
        if args.ppl == 1:
            data.append(eval_data_ppl[i])
        if args.bscore == 1:
            data.append(eval_data_bscore[i])

        eval_data.append(data)
    #print(eval_data)
    #print(train_data)
    data_and_labels = list(zip(train_data, train_labels))
    random.shuffle(data_and_labels)
    train_data, train_labels = zip(*data_and_labels)
    device = torch.device("cuda")
    test_data = torch.tensor(test_data).to(device)
    train_data = torch.tensor(train_data).to(device)
    eval_data = torch.tensor(eval_data).to(device)

    eval_labels = torch.tensor(eval_labels).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    train_labels = torch.tensor(train_labels).to(device)

    model = TextClassifier(num_classes, len(eval_data[0]))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_eval_f1 = 0
    best_model_params = copy.deepcopy(model.state_dict())
    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        epoch_losses = train(model, train_data, train_labels, optimizer, criterion, device, batch_size)
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        eval_labels=eval_labels.to(device)

        eval_predictions,eval_loss = evaluate(model, eval_data,eval_labels, device, batch_size)
        eval_losses.append(sum(eval_loss) / len(eval_loss))
        eval_labels=eval_labels.cpu()
        eval_f1 = f1_score(eval_labels, eval_predictions)

        print(eval_f1)

        if eval_f1 > best_eval_f1:
            print(f"Validation accuracy improved from {best_eval_f1} to {eval_f1}. Saving model...")
            best_eval_f1 = eval_f1
            best_model_params = copy.deepcopy(model.state_dict())
        gc.collect()
        torch.cuda.empty_cache()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('evaluate Loss Over Time')

    model.load_state_dict(best_model_params)
    if args.ppl == 1:
        ppl = "ppl"
    else:
        ppl = ""
    if args.bscore == 1:
        bscore = "bscore"
    else:
        bscore = ""
    torch.save(model.state_dict(), './' + args.model + '_' + ppl + '_' + bscore + 'classifier.pt')
    plt.savefig('./' + args.model + '_' + ppl + '_' + bscore + "_loss.png")
    predictions,_ = evaluate(model, test_data, test_labels,device, batch_size)
    test_labels=test_labels.cpu()
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    print(args.model)
    print("ppl:", args.ppl)
    print("bscore:", args.bscore)
    print("test_Accuracy:", accuracy)
    print("test_Precision", precision)
    print("test_Recall", recall)
    print("test_F1", f1)


if __name__ == "__main__":
    main()
