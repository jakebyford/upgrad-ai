# Create environment
 # python -m venv venv
# source venv/Scripts/activate
# pip install langchain
# pip install langchain-community langchain-openai transformers
# MAC: export OPENAI_API_KEY=<key>
# Windows: set OPENAI_API_KEY=<key>

import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.memory import ConversationBufferMemory


openai_api_key = os.environ.get("OPENAI_API_KEY")
# openai_api_key = "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"

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set")

memory = ConversationBufferMemory(return_messages=True)

conversation_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template = "{history} \n User: {input} \n AI: "
)

conversation_sequence = RunnableSequence(conversation_prompt | conversation_llm)

inputs = [
    "What is Ai?",
    "Can you explain it's applications?",
    "How does it differ from machine learning?"
]

for user_input in inputs:
    full_context = memory.chat_memory.messages if memory else ""
    response_conversation = conversation_sequence.invoke({ "history": full_context, "input": user_input })

    print(f"User: {user_input} \n AI: {response_conversation.content}" )
    memory.save_context({ "input": user_input}, {"output": response_conversation.content})


print("========== JSON FORMAT DEMO ==========")

custom_format_prompt = PromptTemplate(
    input_variables=["topic"],
    template = "Provide the details of {topic} in JSON format with keys : definition, examples and importance",
)

format_sequence = RunnableSequence(custom_format_prompt | conversation_llm)

response = format_sequence.invoke({ "topic": "Natural Language Processing" })

print("========== JSON FORMAT RESPONSE ==========")
print(response.content)