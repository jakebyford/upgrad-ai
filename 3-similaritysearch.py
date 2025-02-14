# Create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
# pip install neo4j torch sentence-transformers matplotlib numpy scikit-learn

import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.WARNING)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

class KnowledgeGraphManager:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()
    
    def create_database(self):
        query = """
            CREATE (p:Person {name: 'Ravi Sharma', description: 'AI researcher focusing on healthcare.'}),
                (f:Field {name: 'AI in Healthcare', description: 'Focuses on AI applications in healthcare.'}),
                (o:Organization {name: 'TechAI Labs', description: 'A leading AI research organization.'}),
                (p2:Person {name: 'Neha Verma', description: 'Data scientist specializing in AI ethics.'}),
                (f2:Field {name: 'AI Ethics', description: 'Explores the ethical implications of AI.'}),
                (pr:Project {name: 'AI-Health', description: 'A project exploring AI applications in disease detection.'})
            WITH p, f, o, p2, f2, pr
            CREATE (p)-[:INTERESTED_IN]->(f),
                (p2)-[:INTERESTED_IN]->(f2),
                (o)-[:WORKS_ON]->(pr)

        """

        try:
            with self.driver.session(database=self.database) as session:
                session.write_transaction(lambda tx: tx.run(query))
        except Exception as e:
            print(e)
    
    def fetch_nodes(self, label="Person"):
        query = f"MATCH (n:{label}) RETURN n.name AS name, n.description AS description"

        try:
            with self.driver.session(database=self.database) as session:
                result = session.read_transaction(lambda tx: list(tx.run(query)))
                return [record.data() for record in result]
        except Exception as e:
            print(e)
            return []
    
    def fetch_relationships(self):
        query = "MATCH (a)-[r]->(b) RETURN a.name AS source, type(r) AS relationship, b.name AS target"

        try:
            with self.driver.session(database=self.database) as session:
                result = session.read_transaction(lambda tx: list(tx.run(query)))
                return [record.data() for record in result]
        except Exception as e:
            print(e)
            return []
    

def generate_embeddings(data, model):
    sentences = [f"{item['name']}: {item['description']}" for item in data]
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    return sentences, embeddings


def perform_similarity_search(embeddings, sentences, query, model, top_k=3):

    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()

    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    top_indices = np.argsort(-similarities[0])[:top_k]
    results = [(sentences[idx], similarities[0][idx]) for idx in top_indices]

    return results


if __name__ == "__main__":
    
    print("Starting knowledge graph embeddings demo ....")
    neo4j = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database="similaritydb")
    #neo4j.create_database()

    data = neo4j.fetch_nodes(label="Person")
    relationships = neo4j.fetch_relationships()

    if not data:
        print("No nodes found in the database.")
    if not relationships:
        print("No relationships found in the database.")

    print("Relationships in the graph")
    for rel in relationships:
        print(f"{rel['source']} -> {rel['relationship']} -> {rel['target']}")
    
    print("Persons in the graph")
    for rel in data:
        print(f"{rel['name']} -> {rel['description']}")
    



    print("Loading Sentence Transformers model ....")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings ....")
    sentences, embeddings = generate_embeddings(data, model)

    query = "AI researcher in healthcare"
    print("Performing similarity search on query: ", query)
    result = perform_similarity_search(embeddings, sentences, query, model)

    for sentence, score in result:
        print(f"Sentence: {sentence}: Score: {score:.2f}")

      
    neo4j.close()