import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
import json
from sklearn.metrics import accuracy_score
import argparse
import gc
import copy
import sys
import argparse
# 步骤2: 定义模型结构

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

        model_output = self.model(input_ids=model_input_ids, attention_mask=model_attention_mask)[1]
        model_output=self.dropout(model_output)
        model_fc1_output=self.fc1(model_output)
        activated_output = self.relu(model_fc1_output)
        output = self.fc2(activated_output)
        return output
def evaluate(model, test_text, model_tokenizer, max_length,device):
    model.eval()
    model_inputs = model_tokenizer(test_text, padding=True, truncation=True, max_length=max_length,
                                     return_tensors='pt')
    model_inputs = model_inputs.to(device)
    with torch.no_grad():
            output = model(model_inputs['input_ids'],model_inputs['attention_mask'])
            softmax = F.softmax(output,dim=-1)

    return  softmax
# 步骤3: 加载模型参数
def roberta_softmax(text):


    model_tokenizer = RobertaTokenizer.from_pretrained('../roberta_base')
    classify_model = RobertaModel.from_pretrained('../roberta_base')
    device = torch.device("cuda")
    model = TextClassifier(classify_model, num_classes=2, input_dim=classify_model.config.hidden_size,type='roberta').to(device)
    model.load_state_dict(torch.load('../three_factors/roberta.pt', map_location=device))
    model.eval()
    max_length = 128  # 与训练时相同的最大长度
    softmax = evaluate(model, text, model_tokenizer, max_length, device)
    return softmax



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='An argparse example')
    parser.add_argument('--text', '-t', default='hello', type=str, help="text")
    args = parser.parse_args()
    
    with open('pre.txt', 'w') as f:
        score = roberta_softmax(args.text)
        f.write(str(score.tolist()))
        

