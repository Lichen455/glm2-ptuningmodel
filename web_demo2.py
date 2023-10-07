import os, sys
import time
import gradio as gr
import mdtex2html

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments


model = None
tokenizer = None

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

global prev_string, count, judgment_End
judgment_End = 0
prev_string=None

def check_and_send(input_string=None):
    # 使用全局变量来跟踪连续相同的字符串和计数
    global prev_string, count, judgment_End
    
    #print(input_string+"xcs")
    if input_string == prev_string:
        count += 1
    else:
        # 如果输入字符串不同，重置计数
        count = 1

    # 更新 prev_string 为当前输入字符串
    prev_string = input_string

    # 如果连续三次输入的字符串相同，执行 send() 函数
    if count == 3:
        send(input_string)
        # 重置计数，以便下次触发
        count = 0
        judgment_End = 1
        return input_string



def send(input_string):
    # 这里可以执行你想要的操作
    #print("连续三次相同，执行 send() 函数")
    print(input_string+"##end")
    #time.sleep(1)
def ax():
    print("hi!")

def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    strs=""
    print(input, chatbot, max_length, top_p, temperature, history, past_key_values)
    chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2 = chatbot, max_length, top_p, temperature, history, past_key_values
    #print(past_key_values2)
    global judgment_End
    # if not isinstance(chatbot, list):
    #     chatbot = [chatbot]
    
    print("User Input:", parse_text(input))
    
    chatbot.append((parse_text(input), ""))
    input = "请出一道编程算法题目，只出题干，不包括解题过程,格式一致"
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature):
        #print(chatbot,"ssssssssssssssssssss")
        if not chatbot:
            chatbot = [('', '')]
        chatbot[-1] = (parse_text(input), parse_text(response))
        #print("Model Response:", parse_text(response))
        strs=check_and_send(parse_text(response))
        #如果检测到生成结束的标记
        if(judgment_End == 1):
            #print(past_key_values)
            chatbot, history, past_key_values = reset_state()
            print("end!x")
            #print(chatbot, max_length, top_p, temperature, history, past_key_values)
            judgment_End = 0
            print(past_key_values2)
            #predict("hello!!!!!", chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2)
            #predict(input, chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2)
            #predict("hello!!!!!", chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2)
            #predict("hello!!!!!", chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2)
            ax()

        yield chatbot, history, past_key_values
        
def predict2(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    strs=""
    print(input, chatbot, max_length, top_p, temperature, history, past_key_values)
    print("执行！")
    chatbot2, max_length2, top_p2, temperature2, history2, past_key_values2 = chatbot, max_length, top_p, temperature, history, past_key_values
    #print(past_key_values2)
    global judgment_End
    # if not isinstance(chatbot, list):
    #     chatbot = [chatbot]
    
    print("User Input:", parse_text(input))
    
    chatbot.append((parse_text(input), ""))
    input = "请出一道编程算法题目，只出题干，不包括解题过程,格式一致"
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                    return_past_key_values=True,
                                                                    max_length=max_length, top_p=top_p,
                                                                    temperature=temperature):
            #print(parse_text(response))
            chatbot[-1] = (parse_text(input), parse_text(response))

            yield parse_text(response)


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM2-6B</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)


def main():
    global model, tokenizer
    
    parser = HfArgumentParser((
        ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    if model_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    model = model.cuda()
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model.transformer.prefix_encoder.float()
    
    model = model.eval()
    
    #生成生成器对象
    prediction_generator = predict2("", [], 8192, 0.8, 0.95, [], None)
    result=""
    try:
        while True:
            #迭代
            result = next(prediction_generator)
            
    except StopIteration:
        # 当生成器迭代完毕时，会引发 StopIteration 异常
        print("生成器迭代完毕")
        print(result)

    #for chat, hist, past_keys in predict2("", [], 8192, 0.8, 0.95, [], None):
        
    print("主函数main运行")
    
    
    #demo.queue().launch(share=False, inbrowser=True)
    
    return

if __name__ == "__main__":
    main()