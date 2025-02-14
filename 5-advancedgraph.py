# Create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
# pip install neo4j matplotlib


import os
from neo4j import GraphDatabase
import logging
import matplotlib.pyplot as plt


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"   


class AdvancedCypherQueriesDemo:

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def create_sample_graph(self):
        queries = [
            """
            CREATE (a:Person {name: 'Alice', age: 30, city: 'Pune'}),
                    (b:Person {name: 'Bob', age: 35, city: 'Mumbai'}),
                    (c:Person {name: 'Charlie', age: 28, city: 'Pune'}),
                    (d:Person {name: 'Diana', age: 25, city: 'Delhi'}),
                    (e:Person {name: 'Eve', age: 29, city: 'Mumbai'}),
                    (p1:Post {content: 'I love AI!', date: '2024-12-01'}),
                    (p2:Post {content: 'Exploring Cypher queries!', date: '2024-12-10'}),
                    (p3:Post {content: 'Graph databases are amazing!', date: '2024-12-05'}),
                    (a)-[:FRIEND]->(b),
                    (a)-[:FRIEND]->(c),
                    (b)-[:FRIEND]->(d),
                    (c)-[:FRIEND]->(e),
                    (a)-[:POSTED]->(p1),
                    (b)-[:POSTED]->(p2),
                    (c)-[:POSTED]->(p3)
            """
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.execute_write(lambda tx: tx.run(query))
            
        print("Sample graph created")

    def find_active_users(self):
        query = """
            MATCH (p:Person)-[:FRIEND]->(f:Person)
            WITH p, COUNT(f) AS friendCount
            WHERE friendCount >= 2
            MATCH (p)-[:POSTED]->(post:Post)
            WHERE post.date >= '2024-12-01'
            RETURN p.name AS name, friendCount, post.content AS recentPost
        """

        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query)))
            return [record.data() for record in result]
    
    def aggreate_age_by_city(self):
        query = """
            MATCH (p:Person)
            RETURN p.city AS city, AVG(p.age) AS averageAge, COUNT(p) AS personCount
        """

        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query)))
            return [record.data() for record in result]
         


if __name__ == "__main__":
    print("Starting Advanced Cypher Queries Demo ....")

    neo4j = AdvancedCypherQueriesDemo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database="advanceddb")


    print("Creating Data in Neo4J database")
    neo4j.create_sample_graph()

    print("Finding Active Users")
    active_users = neo4j.find_active_users()
    for user in active_users:
        print(f"Name: {user['name']}, Friend Count: {user['friendCount']}, Recent Post: {user['recentPost']}")

    
    print("Aggregating Age by City")
    age_data = neo4j.aggreate_age_by_city()
    for city in age_data:
        print(f"City: {city['city']}, Average Age: {city['averageAge']}, Person Count: {city['personCount']}")
    