import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import json
import tqdm
import os


model = "../vicuna/FastChat/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    truncation=True,
)

def Regeneration(question):
    # print("regeneration-----------------------------------------------")
    Regeneration_answers = []
    for i in range(1):
        prompt = "continues the passage from the current text within in total around 300 words:"
        prompt_template = f"### Instruction: {prompt}\n### input: {question}\n### Response:"
        sequences = pipeline(
            prompt_template,
            max_new_tokens=300,
            do_sample=True,
            top_k=10,
            temperature = 0.4,
            num_return_sequences=8,
            eos_token_id=tokenizer.eos_token_id,
        )
        for answer in sequences:

            vicuna_answer = answer["generated_text"].split('### Response:')[-1]
        
            Regeneration_answers.append(vicuna_answer)
    # print("#####################################################################################")
    # print(Regeneration_answers)
    # print("#####################################################################################")
    return Regeneration_answers

if __name__ == "__main__":
    question = "I am sorry to hear that your failed exam"
    answerlist = Regeneration(question)
    # print(answerlist)






