# 
# create environment
#
# pip install langchain-openai faiss-cpu

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = OpenAIEmbeddings()

def stor_embeddings_in_faiss(texts):

    vector_db = FAISS.from_texts(texts, embedding_model)
    return vector_db

def query_faiss_database(vector_db, query_text):
    print(f"Querying the database with: {query_text}")

    results = vector_db.similarity_search(query_text, k=3)
    return results



if __name__ == "__main__":
    print("Starting Vector Database Demo")

    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Deep learning is a subset of machine learning.",
        "Neural networks are inspired by the human brain",
        "AI is used in various industries including healthcare, finance, and education."
    ]

    vector_db = stor_embeddings_in_faiss(texts)

    results = query_faiss_database(vector_db, "What industries uses AI?")

    print(f"Results:")

    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result.page_content}")