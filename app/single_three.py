import torch
import torch.nn as nn
import json
from collections import Counter
import jsonlines
import random
from torch.utils.data import DataLoader, TensorDataset
import logging
import  gc
import random
import copy
import torch.optim as optim
import argparse
import ast

def evaluate(model, test_data, device):
    model.eval()
    test_data = torch.tensor(test_data)
    test_data=test_data.to(device)
    with torch.no_grad():
            output = model(test_data)
            prediction=torch.softmax(output,dim=-1)
    return prediction


class TextClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(TextClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, input_dim *2)
        self.fc2 = nn.Linear(input_dim *2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, data):
        model_output = self.dropout(data)
        model_fc1_output = self.fc1(model_output)
        activated_output = self.relu(model_fc1_output)
        output = self.fc2(activated_output)
        return output

def classify(softmax,ppl,bscore):
    device = torch.device("cpu")
    data=[]
    data.append(softmax[1])
    data.append(ppl)
    data.append(bscore)
    model = TextClassifier(num_classes=2, input_dim=3).to(device)
    model.load_state_dict(
        torch.load('../three_factors/roberta_ppl_bscoreclassifier.pt', map_location=device))
    model.eval()

    prediction = evaluate(model, data, device)
    return prediction

if __name__ == '__main__':
    with open('pre.txt', 'r') as f:
        data = f.read()
        softmax = ast.literal_eval(data)[0]
        print('softmax:', softmax)
    with open('ppl.txt', 'r') as f:
        data = f.read()
        ppl = float("{:.4f}".format(float(data)))
        print('ppl:', ppl)
    with open('Bscore.txt', 'r') as f:
        data = f.read()
        bscore = float("{:.7f}".format(float(data)))
        print('bscore:', bscore)
    
    prediction = classify(softmax, ppl, bscore)
    with open('result.txt', 'w') as f:
        f.write(str(prediction[0].item()))
        print(prediction)
        print(prediction[0].item())