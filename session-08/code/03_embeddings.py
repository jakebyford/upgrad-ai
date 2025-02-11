# Create environment
# pip install langchain openai faiss-cpu matplotlib scikit-learn tiktoken sentence-transformers

import os
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import manifold
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# import scipy
# import numpy
# print("SciPy version:", scipy.__version__)
# print("NumPy version:", numpy.__version__)


def generate_embeddings(text, model = "huggingface"):
    if model == "openai":
        embedding_model = OpenAIEmbeddings()
    elif model == "huggingface":
        # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", token="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")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Invalid model")
    
    return embedding_model.embed_documents(text)

def visualize_embedding(embeddings, text):
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    reduced_embeddings = tsne.fit_transform(np.array(embeddings))

    plt.figure(figsize=(10,3))
    for i, text in enumerate(texts):
        x,y = reduced_embeddings[i]
        plt.scatter(x,y, label=f"Text {i+1}")
        plt.text(x+0.01, y, text[:20], fontsize=9)

    plt.title("Embedding Visualization")
    plt.xlabel = ("Dimension 1")
    plt.ylabel = ("Dimension 2")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Starting Embeddings Demo")

    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Deep learning is a subset of machine learning.",
        "Neural networks are inspired by the human brain"
    ]

    embeddings = generate_embeddings(texts, model="openai")
    print(embeddings)
    print("Generated OpenAI Embeddings")

    visualize_embedding(embeddings, texts)
