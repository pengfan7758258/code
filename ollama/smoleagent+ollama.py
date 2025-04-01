"""
smoleagents: huggingface针对agent做的工具
ollama: 本地部署LLM

特色：
smoleagents有一个Code Agents，生成code来代表agent的action
"""
from smolagents import LiteLLMModel
from smolagents.agents import CodeAgent


model = LiteLLMModel(
    model_id="ollama/deepseek-r1:1.5b",
    api_base="http://192.168.1.55:11434",  # replace with remote open-ai compatible server if necessary
    api_key="ollama",  # replace with API key if necessary
    num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(tools=[], model=model, verbosity_level=2)
respone = agent.run("Could you give me the 10th number in the Fibonacci sequence?")
print(respone)
