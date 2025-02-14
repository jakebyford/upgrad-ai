# Create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
# pip install neo4j openai langchain langchain-community

import os
from neo4j import GraphDatabase
from langchain_community.chat_models import ChatOpenAI

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "basicragdb"



class GraphManager:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {}).data()


graph_manager = GraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)

def create_sample_graph():
    queries = [
        "CREATE (:Topic {name: 'AI in Healthcare', description: 'Applications of AI in diagnostics, treatment, and management.'})",
        "CREATE (:Topic {name: 'AI Ethics', description: 'Ethical considerations and challenges in AI applications.'})",
        "CREATE (:Topic {name: 'AI in Education', description: 'AI for personalized learning and automated grading.'})",
        "MATCH (t1:Topic {name: 'AI in Healthcare'}), (t2:Topic {name: 'AI Ethics'}) CREATE (t1)-[:RELATED_TO]->(t2)",
        "MATCH (t1:Topic {name: 'AI in Education'}), (t2:Topic {name: 'AI Ethics'}) CREATE (t1)-[:RELATED_TO]->(t2)"
    ]

    for query in queries:
        graph_manager.execute_query(query)
    
    print("Sample graph created")


def fetch_graph_data(topic):
    query = """
    MATCH (t:Topic {name: $topic})-[:RELATED_TO]->(related)
    RETURN t.name AS topic, t.description AS description, collect(related.name) AS related_topics
    """

    result = graph_manager.execute_query(query, parameters={"topic": topic})
    return result[0] if result else None



def generate_augmented_response(graph_data, user_query):
    
    context = f"Topic: {graph_data['topic']}\nDescription: {graph_data['description']}\nRelated Topics: {', '.join(graph_data['related_topics'])}"

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    prompt = f"""
    Based on the following context, answer the user's query:

    Context: {context}

    User Query: {user_query}
    """

    return llm.predict(prompt)



if __name__ == "__main__":
    print("Starting Basic RAG with Graph Demo ....")
    create_sample_graph()

    print("Fetching graph data for 'AI in Healthcare'")
    query_topic = "AI in Healthcare"
    graph_data = fetch_graph_data(query_topic)

    if not graph_data:
        print("No data found for the topic")
        exit(1)

    print(f"Result of Graph data for '{query_topic}': {graph_data}")

    user_query = "How is AI tranforming healthcare?"
    response = generate_augmented_response(graph_data, user_query)
    print(f"Response: {response}")


graph_manager.close()