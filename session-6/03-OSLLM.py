# Create environment
 # python -m venv venv
# source venv/Scripts/activate
# pip install langchain
# pip install langchain langchain_openai langchain-community langchain-huggingface transformers accelerate
# MAC: export OPENAI_API_KEY=<key>
# Windows: set OPENAI_API_KEY=<key>

import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import subprocess
import json
from time import sleep

print("========== HuggingFace Inference Engine Demo ==========")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True, device_map="auto")

hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

response = llm.invoke("Explain the benefits of Qwen LLM in Ai applications")

print(f"Response: {response}")





# print("========== Loacl Ollama Inference Engine Demo ==========")
# def ollama_query(model:str, prompt:str):
#     try:
#         subprocess.run(
#             ["ollama", "run", model],
#             input=prompt,
#             text=True,
#             capture_output=True
#         )

#         print(f"Ollam Raw Output: ${output.stdout}")


#     except Exception as e:
#         print(f"Error: {e}")

# ollam_response = ollama_query("qwen2.5-coder:0.5b", "Explain the benefits of Qwen LLM in Ai applications")
