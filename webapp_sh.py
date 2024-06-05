import gradio as gr
from threading import Thread
from single_roberta import roberta_softmax
import subprocess
import time
import os

flag = 1

def init():
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    if os.path.exists("Bscore.txt"):
        os.remove("Bscore.txt")
    if os.path.exists("ppl.txt"):
        os.remove("ppl.txt")
    if os.path.exists("pre.txt"):
        os.remove("pre.txt")
    
  

def wait(file):
    while True:
        time.sleep(1)
        if os.path.exists(file+'.txt'):
            time.sleep(5)
            break

def get_ppl(text):
    subprocess.run(["sbatch", "./sh/run_ppl.sh", text])
    
    

def get_Bscore(text):
    global flag
    if flag == 1:
        subprocess.run(["sbatch", "./sh/run_Bscore.sh", text])
    else:
        with open('Bscore.txt', 'w') as f:
            f.write('0.00001')
            


def get_Pre(text):
    subprocess.run(["sbatch", "./sh/run_pre.sh", text])
    
   
  

def Binary_classification():
    subprocess.run(["sbatch", "./sh/run_classify.sh"])

def run(text):
    
    init()
    score = 0.5
    text = (text,)
    print(type(text))
    t_ppl = Thread(target=get_ppl, args=text)
    t_Bscore = Thread(target=get_Bscore, args=text)
    t_Pre = Thread(target=get_Pre, args=text)

    t_ppl.start()
    t_Bscore.start()
    t_Pre.start()

    t_ppl.join()
    t_Bscore.join()
    t_Pre.join()

    wait('Bscore')
    print('Bscore ok')
    wait('ppl')
    print('ppl ok')
    wait('pre')
    print('pre ok')

    time.sleep(20)
    Binary_classification()

    wait('result')
    
    with open('result.txt', 'r') as f:
        score = float("{:.2f}".format(float(f.read())))
        # print(score)
    label = 'Robot'
    res = {label:(1-score), 'Robot' if label=='Human' else 'Human':score}
    return res

demo = gr.Interface(fn=run, inputs="text", outputs="label")
gr.close_all()
demo.launch()