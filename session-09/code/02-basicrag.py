#
# Create virtual environment
# python -m venv venv
# source venv/bin/activate
# pip install langchain faiss-cpu langchain-openai langchain_community

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def load_documents():
    print("Loading documents...")
    loader = TextLoader("dataset/vector_retriever_sample.txt")
    documents = loader.load()
    return documents

def store_embeddings(documents):
    print("Creating and Storing FAISS Index...")
    vector_db = FAISS.from_documents(documents, embedding_model)
    return vector_db

def query_and_generate(vector_db, query):
    print("Run similarity search and use LLM Retrieval QA Chain...")
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Get embeddings from FAISS --> Search for Similarity --> Get the result, prepare and send to LLM --> Get the response from LLM
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    answer = qa_chain.invoke(query)
    return answer

if __name__ == "__main__":
    print(" Staring Basic RAG Demo ")

    documents = load_documents()

    vector_db = store_embeddings(documents)

    answer = query_and_generate(vector_db, "What does the document say about AI")

    print(f"\n\n Generated Answer: {answer}")

