# Personalized News Aggregator with Knowledge Graph and RAG-based System

This project is a Personalized News Aggregator built using a Retrieval-Augmented Generation (RAG) model, which incorporates a knowledge graph to capture user preferences and interests. The system fetches relevant news articles, uses a language model to generate summaries, and allows users to provide feedback to further personalize their experience.

![News Aggregator](images\images1.png)

![News Aggregator](images\images2.png)

## Features

- **Knowledge Graph**: The system uses a Neo4j-based knowledge graph to store articles, topics, and author information.
- **Document Retrieval**: Articles are retrieved based on user queries, allowing personalized news retrieval.
- **Language Model Integration**: It uses the Gemini API for generative tasks and a pre-trained BART model for summarization.
- **User Feedback**: The app records user feedback on articles, enabling personalized recommendations based on user interactions.

Create a virtual env
``` bash
python - m venv venv
.\venv\Scripts\activate
```
## Requirements
You need to install the following following requirements

```bash
pip install -r requirements.txt
```

### Prerequisites
- Python 3.x
- Neo4j Database
- Google Gemini API Key
- TensorFlow or PyTorch for transformers models

### Usage 
```bash
py - m streamlit run app.py
```

### Approach:

1. **Data Preprocessing**:
   - Crawled 10,000 CC news articles.
   - Collected some news articles using news api.
   - Stored relevant metadata, such as article title, content, authors, and topics in the Neo4j graph.

2. **Graph Construction**:
   - Used Neo4j to create nodes for articles, authors, and topics.
   - Established relationships between these entities (e.g., `WROTE`, `HAS_TOPIC`).

3. **User Interaction**:
   - The user enters a query.
   - The system retrieves relevant articles from the knowledge graph based on the user’s input.
   - A summary of the articles is generated using the language model (facebook/bart-large-cnn)

4. **Personalization**:
   - The system records user feedback (like or dislike) for articles.
   - Based on feedback, the knowledge graph is updated with the user’s preferences.

5. **Output**:
   - A summary of the articles is shown to the user.
   - The user can interact with the system to refine their preferences and get more personalized content.



