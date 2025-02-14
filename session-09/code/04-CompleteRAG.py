# create environment
# python -m venv venv
# source venv\Scripts\activate
# pip install langchain openai pymupdf faiss-cpu wikipedia-api langchain-community langchain-openai

import os
import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

wiki_api = wikipediaapi.Wikipedia(
    language='en',
    user_agent="MyRAGApp/1.0 (https://upgrad.com; contact@upgrad.com)"
)

def load_and_chunk(file_path, chunk_size=500, chunk_overlap=100):
    print("Loading and Splitting the PDF Document...")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunk = text_splitter.split_documents(documents)

    print(f"Number of chunks: {len(chunk)}")
    return chunk

def create_vector_database(chunks):
    print("Creating FAISS Index...")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    print("FAISS Index Created")
    return vector_db

def query_wiki(query):
    print(f"Searching for {query} in Wikipedia...")

    page = wiki_api.page(query)

    if page.exists:
        return page.text[:1000]
    else:
        return "No Wikipedia info found"
    

def structured_response(query, retrievals, wiki_data):

    print("Preparing Structured Response...")

    response_schema = [
        ResponseSchema(name="summary", description="A concise summary of the wikipedia page"),
        ResponseSchema(name="key_points", description="The key points relevant to the query"),
        ResponseSchema(name="wikipedia_reference", description="Relevant information retrieved from Wikipedia")
    ]

    parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = parser.get_format_instructions()

    context = "\n".join([doc.page_content for doc in retrievals])

    response = llm.predict(
        f"""You are an expert assistant. Based on the following context, generate a structured response:
        Context: {context}
        Wikipedia: {wiki_data}
        {format_instructions}"""
    )

    return parser.parse(response)


if __name__ == "__main__":
    print(" Starting Complete RAG Demo ")

    file_path = "./ai-report.pdf"
    chunks = load_and_chunk(file_path)
    vector_db = create_vector_database(chunks)

    queries = [
        "What examples of AI-driven solutions in tutoring are given?"
    ]

    query = queries[0]

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs = {"k": 3}
    )

    retrievals = retriever.get_relevant_documents(query)

    wiki_data = query_wiki(query)

    structured_answer = structured_response(query, retrievals, wiki_data)

    print(f"\n\n Structured Response: *************************")
    print(structured_answer)