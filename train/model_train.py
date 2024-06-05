import torch
import torch.nn as nn
import json
from transformers import BertTokenizer, BertModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from collections import Counter
import jsonlines
import random
from torch.utils.data import DataLoader, TensorDataset
import logging
import gc
import torch.nn.functional as F
import copy
import sentencepiece as spm
import argparse
import matplotlib.pyplot as plt
import os
import sys
def load_jsonl(file_path):
    with open(file_path) as file:
        lines = []
        for line in jsonlines.Reader(file):
            if 'human_answers' in line and line['human_answers']:
                for human_answers in line['human_answers']:
                    line_a = [human_answers, 'human']
                    lines.append(line_a)
            if 'chatgpt_answers' in line and line['chatgpt_answers']:
                for chatgpt_answers in line['chatgpt_answers']:
                    if chatgpt_answers:
                        line_b = [chatgpt_answers, 'chatgpt']
                        lines.append(line_b)
    return lines

def extract_text_and_labels(lines):
    texts = []
    labels = []
    for line in lines:
        text=line[0].lower()
        texts.append(text)
        if line[1] == "human":
            labels.append(0)
        else:
            labels.append(1)

    return texts, labels

def train(model, train_texts, train_labels, optimizer, criterion, model_tokenizer, max_length,batch_size,device):
    model.train()
    num_batches = (len(train_texts) + batch_size - 1) // batch_size
    epoch_losses = []
    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引

        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(train_texts))

        # 获取当前批次的文本和标签
        batch_texts = train_texts[start_index:end_index]
        batch_labels = train_labels[start_index:end_index]

        # 使用 tokenizer 处理当前批次的文本
        model_inputs = model_tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                                     return_tensors='pt')
        model_inputs = model_inputs.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        output = model(model_inputs['input_ids'],model_inputs['attention_mask']).to(device)
        loss = criterion(output, batch_labels)
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return epoch_losses
def evaluate(model, test_texts, test_labels,model_tokenizer, max_length,batch_size,device):
    model.eval()
    predictions = []
    softmax_list=[]
    num_batches = (len(test_texts) + batch_size - 1) // batch_size
    epoch_losses = []
    criterion = nn.CrossEntropyLoss()
    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(test_texts))

        # 获取当前批次的文本和标签
        batch_texts = test_texts[start_index:end_index]
        batch_labels =test_labels[start_index:end_index]
        # 使用 tokenizer 处理当前批次的文本
        model_inputs = model_tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                                     return_tensors='pt')
        model_inputs = model_inputs.to(device)
        with torch.no_grad():
            output = model(model_inputs['input_ids'],model_inputs['attention_mask'])
            loss = criterion(output, batch_labels)
            epoch_losses.append(loss.item())
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.tolist())
    return predictions,epoch_losses




class TextClassifier(nn.Module):
    def __init__(self, model, num_classes, input_dim,type):
        super(TextClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.1)
        self.fc1=nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, num_classes)
        self.relu=nn.ReLU()
        self.type=type
    def forward(self, model_input_ids, model_attention_mask):
        if self.type=='xlnet':
            model_output = self.model(input_ids=model_input_ids, attention_mask=model_attention_mask).last_hidden_state
            model_output = model_output[:, -1, :]
        else:
            model_output = self.model(input_ids=model_input_ids, attention_mask=model_attention_mask)[1]
        model_output=self.dropout(model_output)
        model_fc1_output=self.fc1(model_output)
        activated_output = self.relu(model_fc1_output)
        output = self.fc2(activated_output)
        return output
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='模型的标识或名称')
    args = parser.parse_args()
    if args.model =='bert':
        model_tokenizer = BertTokenizer.from_pretrained('../bert_base_uncased')
        classify_model = BertModel.from_pretrained('../bert_base_uncased')
    else:
        if args.model =='roberta':
            model_tokenizer = RobertaTokenizer.from_pretrained('../roberta_base')
            classify_model = RobertaModel.from_pretrained('../roberta_base')
        else :
            if args.model =='xlnet':
                model_tokenizer = XLNetTokenizer.from_pretrained('../xlnet')
                classify_model = XLNetModel.from_pretrained('../xlnet')
            else:
                sys.exit(1)

    num_classes = 2  # 标签的类别数
    max_length = 128
    epochs = 6

    learning_rate = 1e-5
    batch_size = 4  # 新的批处理大小

    # 初始化模型、优化器和损失函数
    model = TextClassifier(classify_model, num_classes,classify_model.config.hidden_size,args.model)
    logging.info("\nok\n")
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 加载训练集JSONL文件
    train_jsonl_file_path = '../three_data/train.jsonl'
    train_jsonl_data = load_jsonl(train_jsonl_file_path)
    torch.cuda.empty_cache()
    train_texts, train_labels = extract_text_and_labels(train_jsonl_data)
    combined_data = list(zip(train_texts, train_labels))  # 将标签和文本组合成元组列表
    random.shuffle(combined_data)
    train_texts, train_labels = zip(*combined_data)
    train_labels = torch.tensor(train_labels).to(device)

    eval_jsonl_file_path = '../three_data/eval.jsonl'
    eval_jsonl_data = load_jsonl(eval_jsonl_file_path)
    torch.cuda.empty_cache()
    eval_texts, eval_labels = extract_text_and_labels(eval_jsonl_data)
    eval_labels = torch.tensor(eval_labels).to(device)

    test_jsonl_file_path = '../three_data/test.jsonl'
    test_jsonl_data = load_jsonl(test_jsonl_file_path)
    torch.cuda.empty_cache()
    test_texts, test_labels = extract_text_and_labels(test_jsonl_data)
    test_labels = torch.tensor(test_labels).to(device)


    # 训练模型
    best_eval_f1 = 0
    best_model_params = copy.deepcopy(model.state_dict())
    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        epoch_losses = train(model, train_texts, train_labels, optimizer, criterion, model_tokenizer, max_length,
                             batch_size, device)
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        eval_labels=eval_labels.to(device)
        eval_predictions,eval_loss = evaluate(model, eval_texts,eval_labels, model_tokenizer, 128, batch_size, device)
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
    plt.savefig(args.model + "_loss.png")
    # 测试模型
    model.load_state_dict(best_model_params)
    torch.save(model.state_dict(), './'+args.model+'.pt')
    predictions_test, _ = evaluate(model, test_texts,test_labels, model_tokenizer, max_length, batch_size, device)
    test_labels=test_labels.cpu()
    accuracy = accuracy_score(test_labels, predictions_test)
    precision = precision_score(test_labels, predictions_test)
    recall = recall_score(test_labels, predictions_test)
    f1 = f1_score(test_labels, predictions_test)
    print(args.model)
    print("test_Accuracy:", accuracy)
    print("test_Precision",precision)
    print("test_Recall",recall)
    print("test_F1",f1)

if __name__ == "__main__":
  main()

