import os, sys
import time
import gradio as gr
import mdtex2html
import socket

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


prev_string=None



#def send(input_string):

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


def main():
    
    #加载模型
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
    
    
    #网络连接  目标为鸽子一号
    host = '81.71.162.238'  # 服务器的IP地址
    port = 14      # 服务器的端口号

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    while(True):

        #生成生成器对象
        prediction_generator = predict2("", [], 8192, 0.8, 0.95, [], None)

        start_time = time.time()
        result=""
        try:
            while True:
                #迭代
                result = next(prediction_generator)

        except StopIteration:
            #计算生成时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 当生成器迭代完毕时，会引发 StopIteration 异常
            print("生成器迭代完毕")

            print(result,f"  耗时: {elapsed_time:.2f} s")
            time.sleep(0.0001)
            message = result
            client_socket.send(message.encode("utf-8"))

    
    

    print("=====运行结束=====")
    client_socket.close()
    return

if __name__ == "__main__":
    main()