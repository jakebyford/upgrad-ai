# Create python virtual environment 
# pip install virtualenv
# virtualenv venv
# source venv/bin/activate
#  pip install neo4j torch sentence-transformers matplotlib numpy scikit-learn wikipedia-api
# 

import os
import wikipediaapi
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.WARNING)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

wiki_api = wikipediaapi.Wikipedia(language="en", user_agent="DynamicKnowledgeGraphDemo/1.0")


class KnowledgeGraphManager:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def add_wikipedia_data(self, topic):
        page = wiki_api.page(topic)
        if not page.exists():
            print(f"Page {topic} does not exist")
            return
        
        title = page.title
        summary = page.summary
        links = page.links

        with self.driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run("""
                    MERGE (t:Topic {name: $title, description: $summary})
                """, title=title, summary=summary)
            )
        
            for link_title in links.keys():
                session.execute_write(
                    lambda tx: tx.run("""
                        MERGE (lt:Topic {name: $link_title})
                        MERGE (t:Topic {name: $title})
                        MERGE (t)-[:LINKS_TO]->(lt)
                    """, title=title, link_title=link_title)
                )
    
        print(f"Data for Topic '{topic}' added to the knowledge graph")

    def fetch_nodes(self, label="Topic"):
        query = f"MATCH (n:{label}) RETURN n.name AS name, n.description AS description"
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(lambda tx: list(tx.run(query)))
            return [record.data() for record in result]
    
    def fetch_relationships(self):
        """Fetch relationships from the database."""
        query = "MATCH (a)-[r]->(b) RETURN a.name AS source, r.type AS relationship, b.name AS target"
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(lambda tx: list(tx.run(query)))
            return [record.data() for record in result]


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

    neo4j = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database="wikipediadb")

    topics = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]

    print("Calling Wikipedia API to fetch all of the topics ....")
    for topic in topics:
        neo4j.add_wikipedia_data(topic)
    
    print("Fetching nodes from the graph")
    data = neo4j.fetch_nodes(label="Topic")
    relationships = neo4j.fetch_relationships()

    print("Downloading Sentence Transformers model ....")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences, embeddings = generate_embeddings(data, model)

    query = "Applications of Artificial Intelligence in healthcare"
    results = perform_similarity_search(embeddings, sentences, query, model)

    print("============== Similarity Search Results ==============")

    for sentence, score in results:
        print(f"Sentence: {sentence}: Score: {score:.2f}")
      

    # print("============== Relationship Graph ==============")
    # for rel in relationships:
    #     print(f"{rel['source']} -> {rel['relationship']} -> {rel['target']}")


    neo4j.close()