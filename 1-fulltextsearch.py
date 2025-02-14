# Create python virtual environment
# python -m venv venv
# source ./venv/bin/activate
# pip install neo4j  

from neo4j import GraphDatabase


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"


class FullTextSearchManager:

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def create_index(self):
        query = "CREATE FULLTEXT INDEX personIndex IF NOT EXISTS FOR (n:Person) ON EACH [n.name, n.bio]"
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query))

        print("Person Index Created Successfully")    


    def create_relationship_index(self):
        query = "CREATE FULLTEXT INDEX relationshipIndex IF NOT EXISTS FOR ()-[r:KNOWS]-() ON EACH [r.description]"
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query))

        print("Relationship Index Created Successfully")

    def insert_data(self):
        query = """
        CREATE (a:Person {name: 'Alice Smith', bio: 'Data scientist specializing in machine learning'})
        CREATE (b:Person {name: 'Bob Johnson', bio: 'Software engineer with expertise in cloud computing'})
        CREATE (c:Person {name: 'Charlie Brown', bio: 'AI researcher focusing on natural language processing'})
        CREATE (d:Person {name: 'Daisy Ridley', bio: 'Cybersecurity analyst and ethical hacker'})
        CREATE (a)-[:KNOWS {description: 'Colleague in AI research'}]->(c)
        CREATE (b)-[:KNOWS {description: 'Collaborator on cloud projects'}]->(d)
        """
        
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query))
        
        print("Data inserted into the database")
    

    def perform_full_text_search(self, query):

        query = """
        CALL db.index.fulltext.queryNodes('personIndex', $keyword)
        YIELD node, score
        RETURN node.name AS name, node.bio AS bio, score
        ORDER BY score DESC
        """

        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query, keyword=query)))
            return [record.data() for record in result]
    
    def perform_relationship_search(self, query):

        query = """
        CALL db.index.fulltext.queryRelationships('relationshipIndex', $keyword)
        YIELD relationship, score
        RETURN startNode(relationship).name AS source, endNode(relationship).name AS target, relationship.description AS description, score
        ORDER BY score DESC
        """

        with self.driver.session(database=self.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query, keyword=query)))
            return [record.data() for record in result]

if __name__ == "__main__":
    print("Starting Full Text Search Demo ....")

    neo4j = FullTextSearchManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database="fulltextdb")
    
    print("Creating Full Text Indexes with data")
    neo4j.create_index()
    neo4j.create_relationship_index()
    neo4j.insert_data()

    print("Performing Full Text Search on Persons")
    search_keyword = "AI researcher"
    results = neo4j.perform_full_text_search(search_keyword)
    print(results)
    for record in results:
        print(record)
        print(f"Name: {record['name']}, Bio: {record['bio']}, Score: {record['score']}")
    

    print("Performing Full Text Search on Relationships")
    search_keyword = "cloud"
    results = neo4j.perform_relationship_search(search_keyword)
    for record in results:
        print(record)
        print(f"Source: {record['source']}, Target: {record['target']}, Description: {record['description']}, Score: {record['score']}")
    

    neo4j.close()