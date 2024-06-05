import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
import jsonlines
import torch.nn.functional as F
import json
from sklearn.metrics import accuracy_score
import argparse
import gc
import copy
import sentencepiece as spm
import sys
# 步骤2: 定义模型结构
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
def evaluate(model, test_texts, model_tokenizer, max_length,batch_size,device):
    model.eval()
    predictions = []
    softmax_list=[]
    num_batches = (len(test_texts) + batch_size - 1) // batch_size
    for batch_index in range(num_batches):
        # 计算当前批次的起始索引和结束索引
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(test_texts))

        # 获取当前批次的文本和标签
        batch_texts = test_texts[start_index:end_index]

        # 使用 tokenizer 处理当前批次的文本
        model_inputs = model_tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                                     return_tensors='pt')
        model_inputs = model_inputs.to(device)
        with torch.no_grad():
            output = model(model_inputs['input_ids'],model_inputs['attention_mask'])
            softmax = F.softmax(output,dim=-1)
            softmax_list.append(softmax.cpu().numpy())
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.tolist())
    return predictions,softmax_list
# 步骤3: 加载模型参数
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='模型的标识或名称')
    args = parser.parse_args()
    device = torch.device("cuda")
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

    model = TextClassifier(classify_model, num_classes=2, input_dim=classify_model.config.hidden_size,type=args.model).to(device)
    model.load_state_dict(torch.load('../three_factors/'+args.model+'.pt', map_location=device))
    model.eval()

    # 对文本处理quanq
    use_jsonl_file_path = '../three_data/use.jsonl'
    use_jsonl_data = load_jsonl(use_jsonl_file_path)
    torch.cuda.empty_cache()
    use_texts, use_labels = extract_text_and_labels(use_jsonl_data)
    # 加载另一个神经网络测试集JSONL文件
    test_jsonl_file_path = '../three_data/test.jsonl'
    test_jsonl_data = load_jsonl(test_jsonl_file_path)
    torch.cuda.empty_cache()
    test_texts, test_labels = extract_text_and_labels(test_jsonl_data)

    eval_jsonl_file_path = '../three_data/eval.jsonl'
    eval_jsonl_data = load_jsonl(eval_jsonl_file_path)
    torch.cuda.empty_cache()
    eval_texts, eval_labels = extract_text_and_labels(eval_jsonl_data)

    Essay_jsonl_file_path = '../benchmark/Essay_LLMs.jsonl'
    Essay_jsonl_data = load_jsonl(Essay_jsonl_file_path)
    torch.cuda.empty_cache()
    Essay_texts, Essay_labels = extract_text_and_labels(Essay_jsonl_data)

    WP_jsonl_file_path = '../benchmark/WP_LLMs.jsonl'
    WP_jsonl_data = load_jsonl(WP_jsonl_file_path)
    torch.cuda.empty_cache()
    WP_texts, WP_labels = extract_text_and_labels(WP_jsonl_data)

    Reuters_jsonl_file_path = '../benchmark/Reuters_LLMs.jsonl'
    Reuters_jsonl_data = load_jsonl(Reuters_jsonl_file_path)
    torch.cuda.empty_cache()
    Reuters_texts, Reuters_labels = extract_text_and_labels(Reuters_jsonl_data)

    max_length = 128  # 与训练时相同的最大长度
    batch_size = 4
    predictions_use, softmax_list_use = evaluate(model, use_texts, model_tokenizer, max_length, batch_size, device)
    predictions_eval,softmax_list_eval=evaluate(model, eval_texts, model_tokenizer, max_length, batch_size, device)
    predictions_test, softmax_list_test = evaluate(model, test_texts, model_tokenizer, max_length, batch_size, device)
    predictions_Essay, softmax_list_Essay = evaluate(model, Essay_texts, model_tokenizer, max_length, batch_size, device)
    predictions_WP, softmax_list_WP = evaluate(model, WP_texts, model_tokenizer, max_length, batch_size, device)
    predictions_Reuters, softmax_list_Reuters = evaluate(model, Reuters_texts, model_tokenizer, max_length, batch_size, device)

    with open('../three_data/'+args.model+'_softmax_use.jsonl', 'w') as file:
        for sample_softmax in softmax_list_use:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件
    with open('../three_data/' + args.model + '_softmax_eval.jsonl', 'w') as file:
        for sample_softmax in softmax_list_eval:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件

    with open('../three_data/'+args.model+'_softmax_test.jsonl', 'w') as file:
        for sample_softmax in softmax_list_test:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件
    with open('../benchmark/'+args.model+'_softmax_Essay.jsonl', 'w') as file:
        for sample_softmax in softmax_list_Essay:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件
    with open('../benchmark/'+args.model+'_softmax_WP.jsonl', 'w') as file:
        for sample_softmax in softmax_list_WP:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件
    with open('../benchmark/'+args.model+'_softmax_Reuters.jsonl', 'w') as file:
        for sample_softmax in softmax_list_Reuters:
            for softmax in sample_softmax:
                softmax = softmax.tolist()
                file.write(json.dumps(softmax) + '\n')  # 将 logits 转换为 JSON 格式，并写入文件
if __name__ == '__main__':
    main()


