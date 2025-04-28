import threading

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# 设置模型本地路径
model_path = "/home/yiqian-developer/PF/LLaMA-Factory/output/qwen2.5-7b-instruct-lora_20250425"

# 加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
model.eval()

# 单轮对话函数
def chat_single_turn_stream(query, history=None):
    query = query.strip()
    messages = [{"role":"user", "content": query}]
    conversation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # generate时保持与sft的tempalte一致
    inputs = tokenizer(conversation ,return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True # decode不输出特殊template中的字符
    )

    # 用线程运行模型生成，主线流式返回
    generate_kwargs = dict(
        **inputs,
        streamer = streamer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.2
    )
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_response = ""
    for token in streamer:
        partial_response += token
        yield partial_response

# 创建gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_single_turn_stream,
    title="Qwen2.5-7B-Instruct-lora-20250425",
    description="Chat With Model",
    examples=[
        "我有10个两分的和五分的硬币,共35分,请问这两种硬币各有多少个?", 
        "我在池塘养了两种动物，分别是螃蟹和鸭子。它们合起来有8个头，有40只脚。请问螃蟹和鸭子各几只？"
    ],
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue")
)
# 启动服务
demo.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=7860
)