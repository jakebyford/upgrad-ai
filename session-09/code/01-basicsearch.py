# create environment
# python -m venv venv
# source venv\Scripts\activate
# pip install langchain faiss-cpu langchain-openai

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import retrieval_qa
from langchain_community.chat_models import ChatOpenAI

embedding_model = OpenAIEmbeddings()
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

import os

def fetch_and_load_documents():
    dataset_path = "dataset/vector_retriever_sample.txt"
    
    if not os.path.exists(dataset_path):
        os.makedirs("dataset", exist_ok=True)

        with open(dataset_path, "w") as f:
            f.write("\n".join([
                "AI is transforming industries like healthcare, finance, and education.",
                "Artificial Intelligence helps automate repetitive tasks.",
                "AI applications include natural language processing and computer vision.",
                "Deep learning has advanced AI significantly in recent years.",
                "AI raises ethical concerns, including bias and privacy issues."
            ]) + "\n")  # Ensures a newline at the end
    
    print(f"Dataset saved at {dataset_path}")

    loader = TextLoader(dataset_path)
    documents = loader.load()
    return documents


def create_faiss_index(documents):
    print("Creating FAISS Index...")
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local("faiss_indexes/vector_store_retriver")
    print("FAISS Index created and saved locally")
    return vector_db


def similarity_search(vector_db, query, k=3):
    print("Performing similarity search...")
    similar_docs = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k}).get_relevant_documents(query)
    return similar_docs


if __name__ == "__main__":
    print("Starting Basic RAG Demo")

    documents = fetch_and_load_documents()

    vector_db = create_faiss_index(documents)

    query = "How does AI help industries?"

    similar_docs = similarity_search(vector_db, query, k=3)
    
    for i, doc in enumerate(similar_docs):
        print(f"Result {i+1}: {doc.page_content}")