import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def fetch_and_load_documents():
    """Fetch and load a conversational dataset."""
    print("\n> Fetching dataset...")
    dataset_path = "dataset/conversational_faq_sample.txt"
    if not os.path.exists(dataset_path):
        os.makedirs("dataset", exist_ok=True)
        with open(dataset_path, "w") as f:
            f.write(
                "Q: What is AI?\n"
                "A: AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines.\n"
                "Q: What are some applications of AI?\n"
                "A: Applications include natural language processing, computer vision, and robotics.\n"
                "Q: What industries benefit from AI?\n"
                "A: Industries like healthcare, finance, and education benefit greatly from AI.\n"
                "Q: What are ethical concerns in AI?\n"
                "A: Ethical concerns include bias, privacy, and decision-making transparency."
            )
    print(f"Dataset loaded from: {dataset_path}")
    loader = TextLoader(dataset_path)
    documents = loader.load()
    return documents

def create_faiss_index(documents):
    print("Creating FAISS index...")

    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local("vector_db")
    return vector_db

def conversational_retriever(vector_db, query, search_type="similarity", k=3, threshold=None):
    print("Performing Conversational Retrieval...")

    memory = ConversationBufferMemory(memory_type="chat_history", return_messages=True)

    search_kwargs = {"k": k}

    if search_type == "similarity_score_threshold":
        search_kwargs['score_threshold'] = threshold or 0.5

    retriever = vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)

    answer = retrieval_chain.invoke({"query":query})

    return answer

if __name__ == "__main__":
    print(" Starting the Conversational retrieval demo ")

    documents = fetch_and_load_documents()

    vector_db = create_faiss_index(documents)

    query = "Tell me about AI applications."

    #similarity search demo
    answer = conversational_retriever(vector_db, query, search_type="similarity", k=3)
    print(f"SIMILARITY SEARCH : {answer}")

    #similarity score search demo
    answer = conversational_retriever(vector_db, query, search_type="similarity_score_threshold", k=3, threshold=0.6)
    print(f"SIMILARITY THRESHOLD SEARCH : {answer}")

    # Using Maximal Marginal Relevance (MMR) search demo
    answer = conversational_retriever(vector_db, query, search_type="mmr", k=3)
    print(f"Maximal Marginal Relevance (MMR) SEARCH : {answer}")