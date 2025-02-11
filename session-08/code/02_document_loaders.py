# environment creation
# python -m venv venv
# source venv/bin/activate
# pip install -U langchain langchain_community pdfminer.six beautifulsoup4 pypdf unstructured langchain_openai

import os
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredURLLoader


def load_text_file(file_path):
    
    loader = TextLoader(file_path)
    docs = loader.load()
    print(f"====== Text Document Loaded {file_path} =======")
    for doc in docs:
        
        print(doc.page_content)

def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"====== PDF Document Loaded {file_path} =======")
    for doc in docs:
        print(doc.page_content)

def load_csv_file(file_path):
    loader = CSVLoader(file_path)
    docs = loader.load()
    print(f"====== CSV Document Loaded {file_path} =======")
    for doc in docs:
        
        print(doc.page_content)

def load_url(url):
    loader = UnstructuredURLLoader(url)
    docs = loader.load()
    print(f"====== URL Crawler Loaded {url} =======")
    for doc in docs:
        print(doc.page_content)

if __name__ == "__main__":
    print("Starting Document Loaders Demo")

    text_file = "../../sample3.txt"
    pdf_file = "../../resume_JakeByford_Tex.pdf"
    csv_file = "../../country_full.csv"
    url = "https://en.wikipedia.org/wiki/2024_United_States_presidential_election"

    load_text_file(text_file)
    load_pdf_file(pdf_file)
    # load_csv_file(csv_file)
    load_url(url)


        
## lot more loaders in langchain
