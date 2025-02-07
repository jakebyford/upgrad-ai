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

# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = "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"

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

prompt = PromptTemplate(
    input_variables=["name", "topic"],
    template = "Hi {name} , Explain the concept of {topic} in simple terms."
)

sequence = RunnableSequence(prompt | llm)

if __name__ == "__main__":
    topic = "quantum computing"
    response = sequence.invoke({ "name": "Jake", "topic":topic })
    print(f"Topic: {topic} \n Response: {response.content}" )

    try:
        response = sequence.invoke({ "topic":topic })
        print(f"Topic: {topic} \n Response: {response.content}" )
    except Exception as e:
        print(f"Error: {e}")

    def generate_prompt(topic, name="User"):
        reusable_prompt = PromptTemplate(
            input_variables=["name", "topic"],
            template = "Hi {name} , Explain the concept of {topic} in simple terms."
        )

        resusable_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
        reusable_sewquence = RunnableSequence(reusable_prompt |resusable_llm)
        return reusable_sewquence.invoke({ "name": name, "topic":topic })
    
    print("========== REUSABLE DEMO ==========")
    print(generate_prompt("quantum computing", "Jake"))

    print("========== TOKEN AND TRUNCATE DEMO ==========")

    promt_truncate = PromptTemplate(
        input_variables=["topic"],
        template = "Explain the concept of {topic} in simple terms.",
    )

    llm_truncate = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key, max_tokens=50)

    sequence_truncate = RunnableSequence(promt_truncate | llm_truncate)

    response = sequence_truncate.invoke({ "topic": "quantum computing" })
    print(response.content)