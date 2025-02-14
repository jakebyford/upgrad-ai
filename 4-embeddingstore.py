# create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
# pip install neo4j openai langchain langchain-community scikit-learn tiktoken


import os
import numpy as np
from neo4j import GraphDatabase
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "embeddingstoredb"

class Neo4jEmbeddingManager:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def create_sample_data(self):

        query = """
        CREATE (a:Domain {name: 'AI in Healthcare', description: 'AI applications in diagnostics and treatment'}),
               (b:Domain {name: 'AI in Education', description: 'AI enhances personalized learning'}),
               (c:Domain {name: 'AI in Finance', description: 'AI used in fraud detection and trading'}),
               (d:Domain {name: 'AI in Transportation', description: 'AI enables autonomous vehicles'}),
               (a)-[:RELATED]->(b),
               (b)-[:RELATED]->(c),
               (c)-[:RELATED]->(d)
        """

        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query))

        print("Sample Graph data created")

    def fetch_nodes(self):
        query = "MATCH (n:Domain) RETURN n.name AS name, n.description AS description, id(n) AS id"
        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query)))
            return [{"id": record["id"], "name": record["name"], "description": record["description"]} for record in result]

    def update_node_embeddings(self, node_id, embeddings):
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        SET n.embedding = $embedding
        """

        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, node_id=node_id, embedding=embeddings))
    

    def fetch_node_embeddings(self):
        query = """
        MATCH (n:Domain) 
        WHERE n.embedding IS NOT NULL 
        RETURN n.name AS name, n.embedding AS embedding
        """

        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query)))
            return [record.data() for record in result]



def generate_embeddings(nodes, embedding_model):
    
    description = [ node["description"] for node in nodes ]
    embeddings = embedding_model.embed_documents(description)
    return embeddings


def perform_similarity_search(embeddings, query_embedding, nodes, top_k=3):
    """Perform similarity search using cosine similarity."""
    similarities = cosine_similarity([query_embedding], embeddings)
    top_indices = np.argsort(-similarities[0])[:top_k]
    results = [(nodes[idx]["name"], similarities[0][idx]) for idx in top_indices]
    return results



if __name__ == "__main__":
    print("Starting Embedding Store Demo ....")

    db_manager = Neo4jEmbeddingManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)
    db_manager.create_sample_data()

    nodes = db_manager.fetch_nodes()

    print("Generating embeddings for the nodes")
    embedding_model = OpenAIEmbeddings()
    embeddings = generate_embeddings(nodes, embedding_model)

    print("Update embeddings in the database")
    for i, node in enumerate(nodes):
        db_manager.update_node_embeddings(node["id"], embeddings[i])
    
    print("Embeddings stored int he database")

    node_embeddings = db_manager.fetch_node_embeddings()
    embeddings = np.array([node["embedding"] for node in node_embeddings])
    node_names = [{"name": item["name"]} for item in node_embeddings]

    print("Performing similarity search on the embeddings")
    query = "AI used for personalized learning"
    query_embedding = embedding_model.embed_query(query)

    results = perform_similarity_search(embeddings, query_embedding, node_names)
    for name, score in results:
        print(f"Node: {name}, Score: {score:.2f}")



    db_manager.close()