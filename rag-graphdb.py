# create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
#  pip install neo4j torch langchain matplotlib numpy openai langchain_community scikit-learn tiktoken


import os
from neo4j import GraphDatabase
from langchain_community.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "embeddingsrag2"

class Neo4JManager:

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
    def close(self):
        self.driver.close()
    
    def create_sample_data(self):
        query = """
        CREATE (n1:Domain {name: 'AI in Healthcare', description: 'AI transforms healthcare with personalized treatments.'}),
        (n2:Domain {name: 'AI in Finance', description: 'AI detects fraud and automates trading.'}),
        (n3:Domain {name: 'AI in Education', description: 'AI enables adaptive learning experiences.'}),
        (n4:Domain {name: 'AI in Transportation', description: 'AI powers autonomous vehicles.'})
        """

        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query))

        print("Sample Graph data created")

    def fetch_embeddings(self):
        query = """
            MATCH (n:Domain)
            WHERE n.embedding IS NOT NULL
            RETURN n.name AS name, n.embedding AS embedding
        """
        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query)))    
            return [{"name": record["name"], "embedding": np.array(record["embedding"])} for record in result]


    def store_embeddings(self, nodes, embeddings):
        query = """
            MATCH (n {name: $name})
            SET n.embedding = $embedding
        """

        with self.driver.session(database=self.database) as session:
            for node, embedding in zip(nodes, embeddings):
                session.execute_write(lambda tx: tx.run(query, name=node["name"], embedding=embedding))




def generate_embeddings(data, model):
    description = [item['description'] for item in data]
    embeddings = model.embed_documents(description)
    return embeddings



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def perform_similarity_search(embeddings, query_embedding):

    print("************ EMBEDDINGS")
    print (embeddings)

    query_embedding = np.array(query_embedding)

    embedding_matrix = np.stack([np.array(e["embedding"]) for e in embeddings])

    similarities = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)

    top_indices = np.argsort(-similarities[0])[:3]

    results = [
        {
            "node": embeddings[i],
            "score": similarities[0][i] 
        } for i in top_indices
    ]

    return results



if __name__ == "__main__":
    print("Starting RAG Demo ....")
    
    db_manager = Neo4JManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)

    db_manager.create_sample_data()

    sample_data = [
        {"name": "AI in Healthcare", "description": "AI transforms healthcare with personalized treatments."},
        {"name": "AI in Finance", "description": "AI detects fraud and automates trading."},
        {"name": "AI in Education", "description": "AI enables adaptive learning experiences."},
        {"name": "AI in Transportation", "description": "AI powers autonomous vehicles."},
    ]

    from langchain_community.embeddings import OpenAIEmbeddings

    embedding_model = OpenAIEmbeddings()
    embeddings = generate_embeddings(sample_data, embedding_model)
    db_manager.store_embeddings(sample_data, embeddings)

    node_embeddings = db_manager.fetch_embeddings()

    query = "How is AI used in healthcare?"
    query_embedding = embedding_model.embed_query(query)

    search_results = perform_similarity_search(node_embeddings, query_embedding)

    print("Similarity Search Results:")
    for result in search_results:   
        node = result["node"]
        score = result["score"]
        print(f"Node: {node['name']}, Score: {score:.2f}")

    context = ". ".join([
        f"{result['node']['name']} - {node['description']}"
        for result in search_results
        for node in sample_data
        if node["name"] == result["node"]["name"]
    ])
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    prompt = f"Context: {context} \n\n Query: {query} \n\nAnswer:"

    response = llm.predict(prompt)


    print("********** LLM Response:")
    print(response)

    db_manager.close()

