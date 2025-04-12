"""
openai的batch api接口
地址：https://platform.openai.com/docs/guides/batch?lang=python

下面技巧可以列出所有提交过的batch任务：
import openai
# 列出所有批处理任务
batches = openai.batches.list()

batches.data
"""

import json
import time
import yaml

import pandas as pd
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

class OpenaiBatchRequest:

    def __init__(self, 
                 prompt_file: str,
                 ) -> None:
        """
        example:
        # 1.创建对象
        obr = OpenaiBatchRequest(
            whwm_prompt_file,
        )
        # 2.构建jsonl上传数据集
        questions = [q1,q2...]
        answers = [a1,a2...]
        obr.create_openai_batch_jsonl(
            "数据集来源描述", 
            "openaiBatchInput.jsonl", # jsonl文件名
            questions, 
            answers)
        # 3.上传jsonl并开启batch任务
        obr.create_batch_task()
        # 4.保存对话数据到csv
        while True:
            if obr.task_status == "in_progress":
                time.sleep(5)
            elif obr.task_status == "completed":
                obr.to_csv(path="data.csv")
                break
            else:
                print("有问题, task status：", obr.task_status)
                break
        """
        self.prompt_file = prompt_file

        self.client = OpenAI()
        self.chat_prompts = self._load_chat_prompts()

    @property
    def task_status(self):
        return self.client.batches.retrieve(self.batches_id).status

    def _load_chat_prompts(self):
        with open(self.prompt_file, 'r') as f:
            prompts = yaml.safe_load(f)

        system_prompt = prompts.get('system_prompt', '').replace('{', '{{').replace('}', '}}')
        
        character_prompt = "还有没有拼接的system prompt信息"

        system_prompt += '\n' + character_prompt

        few_shot_examples = [
            {
                'question': case['question'].replace('{', '{{').replace('}', '}}'),
                'answer': case['answer'].replace('{', '{{').replace('}', '}}')
            }
            for case in prompts.get('examples', [])
        ]

        prompt_messages = [
            ('user', '{question}')
        ]

        prompt_messages.insert(0, ("system", system_prompt)) # 将system message插入首位

        # 构建few shot eample massagess
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ('user', '{question}'),
                ('assistant', '{answer}')
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shot_examples
        )
        prompt_messages.insert(1, few_shot_prompt)

        return ChatPromptTemplate.from_messages(prompt_messages)
    
    def roleName_replaced(self, msg_type: str):
        if msg_type == "human":
            return "user"
        if msg_type == "ai":
            return "assistant"
        if msg_type == "system":
            return "system" 
        
    def create_openai_batch_jsonl(self, 
                                source_description: str,
                                batch_jsonl_path: str,
                                questions: list,
                                answers: list,
                                model_name="gpt-4o"):
        self.source_description = source_description
        self.questions = questions
        self.answers = answers
        self.batch_jsonl_path = batch_jsonl_path
        print("开始构建openai batch jsonl输入数据:", self.batch_jsonl_path)
        with open(self.batch_jsonl_path, "w+", encoding="utf-8") as f:
            for request_id, question in enumerate(questions):
                # 渲染 Prompt
                messages = self.chat_prompts.format_messages(question=question)

                # 转换成 openai batch 中要求的jsonl结构
                json_messages = [{"role": self.roleName_replaced(msg.type), "content": msg.content} for msg in messages]
                
                final_payload = {
                    "custom_id": f"request-{request_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": f"{model_name}",
                        "messages": json_messages
                    }
                }
                f.write(json.dumps(final_payload, ensure_ascii=False)+"\n")
        print("构建成功")

    def create_batch_task(self):
        print("开始上传jsonl文件:", self.batch_jsonl_path)
        batch_input_file = self.client.files.create(
            file=open(self.batch_jsonl_path, "rb"),
            purpose="batch"
        )
        print("jsonl文件上传成功")

        batch_input_file_id = batch_input_file.id
        # print("batch_input_file_id:", batch_input_file_id)

        print("创建batch任务~")
        batches = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print("batch任务创建成功~")
        self.batches_id = batches.id

    def monitor_batch_task(self):
        print("开始监控任务:", self.batches_id)
        
        try:
            print("选择接下来的监控:")
            print("1.按键p：查看task状态")
            print("2.按键q：每5秒输出一次task状态，直到completed")
            choice = input("请输入 p 或 q：").strip().lower()

            batch = self.client.batches.retrieve(self.batches_id)
            task_status = batch.status
            if choice == 'p':
                print("当前task状态：", task_status)
                time.sleep(5)
            elif choice == 'q':
                while True:
                    task_status = self.client.batches.retrieve(self.batches_id).status
                    print("5秒刷新一次task状态,当前task状态：", task_status)
                    # 检查是否退出
                    if task_status == 'completed':
                        break
                    time.sleep(5)
            else:
                print("无效输入，请输入 'p' 或 'q'。")
        except Exception as e:
            print(f"error: {e}")

    def to_csv(self, 
               path: str,
               columns: list[str]=["source_description", "who_answer", "question", "true_answer", "predict_answer", "correction_answer"]
               ):
        """
        结果存储csv

        path: 保存路径
        columns: 存储的字段名
        预设字段：source_description(源数据描述), who_answer, question, answer, correction_answer(校正答案) -> 校正部分的answer和correction_answer使用<correct></correct>标签包裹
        """
        task_status = self.client.batches.retrieve(self.batches_id).status
        if task_status != "completed":
            print(f"当前的task状态：{task_status},无法执行")
            return


        # 获得结果对象
        file_response = self.client.files.content(self.client.batches.retrieve(self.batches_id).output_file_id)

        # 用列表存储所有数据
        all_data = []
        # 写入文件
        for question, t_answer, resp in zip(self.questions, self.answers, file_response.iter_lines()):
            p_answer = json.loads(resp)["response"]["body"]["choices"][0]["message"]["content"]
            row = {
                "source_description": self.source_description,
                "who_answer": "gpt-4o",
                "question": question,
                "true_answer": t_answer,
                "predict_answer": p_answer,
                "correction_answer": "" # 人工
            }
            all_data.append(row)

        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(path, index=False, encoding="utf-8")
