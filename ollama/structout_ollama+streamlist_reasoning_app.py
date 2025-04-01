"""
streamlit+ollama结构化输出
"""

import os
import time
from typing import Literal

import ollama
from pydantic import BaseModel
import streamlit as st

client = ollama.Client(host="http://192.168.1.56:11434")

class ReasoningStep(BaseModel):
    title: str
    content: str
    next_action: Literal["continue", "final_answer"]

class FinalAnswer(BaseModel):
    title: str
    content: str

def make_api_call(messages, max_tokens, is_final_answer=False):
    """
    调用ollama api
    messages: list 聊天列表
    max_tokens: int 模型最大生成token数
    is_final_answer: bool 是否是最终答案
    """
    # 三次机会重试，可以调整
    for attempt in range(3):
        try:
            format_schema = ReasoningStep if not is_final_answer else FinalAnswer
            response = client.chat(
                model="deepseek-r1:1.5b",
                messages=messages,
                options={"temperature": 0.2, "num_predict": max_tokens},
                format=format_schema.model_json_schema(),
            )
            print("ollama response:"+"-"*30)
            print(response)
            print("-"*30)
            return format_schema.model_validate_json(response.message.content) # .model_validate_json接收一个json字符串
        except Exception as e:
            if attempt == 2:
                if is_final_answer: # 如果是最终回答
                    return FinalAnswer(title="Error", content=f"Failed to generate final answer after 3 attempts. Error: {str(e)}")
                else: # 如果还没到最终回答，还在中间步骤
                    return ReasoningStep(title="Error", 
                                         content=f"Failed to generate step after 3 attempts. Error: {str(e)}", next_action="final_answer")
            time.sleep(1)  # 重试前停顿1秒

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data.title}", step_data.content, thinking_time))
        
        messages.append({"role": "assistant", "content": step_data.model_dump_json()}) # .model_dump_json转换为json字符串
        
        if step_data.next_action == 'final_answer' or step_count > 25: # 最多25步，防止无限思考时间。可以调整.
            break
        
        step_count += 1

        # Yield Streamlit更新每一个step
        yield steps, None  # 我们要到最后才公布总时间

    for msg in messages:
        print(msg['role'], msg['content'][:50])
        
    # 生成final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data.content, thinking_time))

    yield steps, total_thinking_time

def main():
    # 页面配置
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
    
    st.title("g1: 使用deepseek-r1-distll:1.5b在本地创建类似于o1的推理链")
    
    st.markdown("""
    这是使用prompt创建类似o1推理链以提高输出准确性的早期prototype。它并不完美，准确性尚未得到正式评估。它由Ollama提供动力.
                
    Open source [仓库地址](https://github.com/bklieger-groq)
    """)
    
    # 用户输入
    user_query = st.text_input("用户输入:", placeholder="e.g., How many 'R's are in the word strawberry?")
    
    if user_query:
        st.write("生成响应中...")
        
        # 创建empty elements 占位generated text和total time
        response_container = st.empty()
        time_container = st.empty()
        
        # 生成并展示response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            
            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

if __name__ == "__main__":
    main()