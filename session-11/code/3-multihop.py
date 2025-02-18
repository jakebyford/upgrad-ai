query = """
        CREATE (ai:Technology {name: 'AI', description: 'Artificial Intelligence'})
        CREATE (healthcare:Industry {name: 'Healthcare', description: 'Applications in medical and health fields'})
        CREATE (diagnostics:Application {name: 'Diagnostics', description: 'AI in disease detection'})
        CREATE (radiology:SubApplication {name: 'Radiology', description: 'AI in radiology diagnostics'})
        CREATE (treatment:Application {name: 'Treatment', description: 'AI in personalized treatments'})
        CREATE (education:Industry {name: 'Education', description: 'Applications in teaching and learning'})
        CREATE (elearning:Application {name: 'E-Learning', description: 'AI in online education'})

        
        MERGE (ai)-[:IMPACTS]->(healthcare)
        MERGE (ai)-[:IMPACTS]->(education)
        MERGE (healthcare)-[:HAS_APPLICATION]->(diagnostics)
        MERGE (diagnostics)-[:HAS_SUBAPPLICATION]->(radiology)
        MERGE (healthcare)-[:HAS_APPLICATION]->(treatment)
        MERGE (education)-[:HAS_APPLICATION]->(elearning)
        """

cypher_query = """
    MATCH (a)-[r]->(b)-[r2]->(c)
    RETURN a.name AS source, type(r) AS relationship, b.name AS intermediate, c.name AS target
    """