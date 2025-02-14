# py -m streamlit run app.py
import streamlit as st
from py2neo import Graph, Node, Relationship
from transformers import pipeline
import json
import google.generativeai as genai
from elasticsearch import Elasticsearch
import numpy as np

GEMINI_API_KEY = "AIzaSyCEYlFHaxn2vZMvivXGtHnc5Rglg9YYeV4"
NEO4J_URI="neo4j+s://4b1bd62f.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="_v7r2jzEUO39ynCedYsMFs5CHYpGAYFx9D1DwsDVtiU"
CLOUD_ID = 'Anshika_News_Aggregator:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbyRmZDgzYmIyZGYwNzY0MjE1ODczM2U5MWNlZmZjMjhlOSQxYjNmNmY2YTk2ODI0NWI2OWYwYmM5ZTllY2VmMDk5ZQ=='
USERNAME = 'elastic'
PASSWORD= '3FYsRWd1MygcQZgX5Gu25EGZ'
INDEX_NAME = "all_news_article"

# Neo4j Connection
graph = Graph(
    NEO4J_URI,  
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD) 
)

es = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(USERNAME, PASSWORD)
)

# Language Model 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 1: Populate the Knowledge Graph
def populate_knowledge_graph(graph, data):
    """
    Populates the Neo4j knowledge graph with articles, topics, and authors from the dataset.
    """
    for article in data:
        # Add Article Node
        article_node = Node(
            "Article",
            title=article["title"],
            text=article["text"],
            publishing_date=article["publishing_date"]
        )
        graph.merge(article_node, "Article", "title")
        #  Add Embedding (from Elasticsearch)
        embedding = retrieve_embedding_from_elasticsearch(article["title"],article["text"])
        if embedding:
            article_node["embedding"] = embedding.tolist()
            graph.push(article_node)

        # Add Author Nodes and Relationships
        for author in article.get("authors", []):
            author_node = Node("Author", name=author)
            graph.merge(author_node, "Author", "name")
            graph.merge(Relationship(author_node, "WROTE", article_node))

        # Add Topic Nodes and Relationships
        for topic in article.get("topics", []):
            topic_node = Node("Topic", name=topic)
            graph.merge(topic_node, "Topic", "name")
            graph.merge(Relationship(article_node, "HAS_TOPIC", topic_node))

def retrieve_embedding_from_elasticsearch(title, text):
    """
    Retrieves embeddings from Elasticsearch based on the article title and text.
    """
    try:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": title}},
                        {"match": {"text": text}}
                    ]
                }
            },
            "_source": ["embedding"]
        }
        # Replace 'your_index_name' with the actual Elasticsearch index name
        response = es.search(index=INDEX_NAME, body=query, size=1)
        
        if response["hits"]["hits"]:
            # Return the first matching embedding
            return np.array(response["hits"]["hits"][0]["_source"]["embedding"])
        else:
            print("No matching embedding found.")
            return None
    except Exception as e:
        print(f"Error retrieving embedding: {e}")
        return None

# Step 2: Retrieve Relevant Context from Knowledge Graph
def retrieve_context(graph, user_query):
    """
    Retrieves articles and topics relevant to the user's query from the knowledge graph.
    """
    query = """
    MATCH (t:Topic)<-[:HAS_TOPIC]-(a:Article)
    WHERE t.name CONTAINS $topic
    RETURN a.title AS title, a.text AS text
    """
    result = graph.run(query, topic=user_query).data()
    return result

# Step 3: Generate a Response using Language Model
def generate_response(context, user_query):
    """
    Generates a response using the retrieved context and the user's query.
    """
    context_text = "\n".join([f"{item['title']}: {item['text']}" for item in context])
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Step 4: Update User Preferences in Knowledge Graph
def update_preferences(graph, user_name, article_title, action):
    """
    Updates the knowledge graph with user preferences (LIKE or DISLIKE).
    """
    user_node = Node("User", name=user_name)
    article_node = graph.nodes.match("Article", title=article_title).first()
    if action == "LIKE":
        graph.merge(Relationship(user_node, "LIKES", article_node))
    elif action == "DISLIKE":
        graph.merge(Relationship(user_node, "DISLIKES", article_node))

def main():
    st.title("Personalized News Aggregator📰")
    st.sidebar.header("Features")
    st.sidebar.markdown("""
    - Knowledge Graph for Preferences
    - Document Retrieval
    - News Summarization
    - Personalized Responses
    """)

    # Load Dataset
    # with open('preprocessed_news_articles.json', 'r',encoding='utf-8') as file:
    #     dataset = json.load(file)

    # Populate the Knowledge Graph
    st.write("Populating the Knowledge Graph...")
    # populate_knowledge_graph(graph, dataset)
    st.write("Knowledge Graph populated successfully.")

    user_query = st.text_input("Enter your query:")

    if user_query:
        # Retrieve context from the graph
        context = retrieve_context(graph, user_query)

        if not context:
            st.write("No relevant information found.")
        else:
            st.write("Relevant articles:")
            for item in context:
                st.write(f"**Text**: {item['text'][:310]}...") 
                st.write("---")

            # Generate a response
            response = generate_response(context, user_query)
            print(response)
            st.write("### Generated Response:")
            st.write(response)

            if st.button("Summarize"):
                summary = summarizer(response, max_length=200, do_sample=False)
                st.write("### Summary:")
                st.write(summary[0]["summary_text"])
            
            feedback = st.radio("Was this information helpful?", ("LIKE", "DISLIKE"),index=0)

            if feedback:
                article_title = context[0]["title"]
                st.write("Feedback recorded. Thank you!")

if __name__ == "__main__":
    main()
