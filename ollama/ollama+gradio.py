"""
ollama
gradio
快速搭建一个可以聊天的界面
"""

import os 
import gradio as gr
from openai import OpenAI

# 配置openai client
client = OpenAI(
    base_url="http://192.168.1.55:11434/v1/",
    api_key="ollama"
)

def chat_with_bot(message, history):
    """
    """
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})
    # print("------messages----")
    # print(messages)
    # print("----------")
    try:
        response = client.chat.completions.create(
            model="deepseek-r1:1.5b",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        bot_response = response.choices[0].message.content
        # print("-----response-----")
        # print(bot_response)
        # print("----------")

        # bot_response = input("请输入：")
        return bot_response
    except Exception as e:
        return f"Error: {e}"

# 流式输出
def chat_with_bot_stream(message, history):
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    try:
        # 使用stream=True启用流式响应
        stream = client.chat.completions.create(
            model="deepseek-r1:1.5b",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )
        
        # 逐步返回响应内容
        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message
                
    except Exception as e:
        yield f"Error: {e}"



# 创建gradio chat interface
demo = gr.ChatInterface(
    fn=chat_with_bot,
    title="Deepseek-r1:1.5b",
    description="Chat with Deepseek-R1:1.5b model",
    examples=["Hello!How are you?",
              "Can you help me with python programming?",
              "Tell me a short story"],
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green"
    ),
    escape_html=False
)

demo.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=7860
)