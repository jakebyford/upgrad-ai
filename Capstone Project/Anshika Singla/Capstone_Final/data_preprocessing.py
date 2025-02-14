import json
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Preprocessing function
def preprocess_json(data):
    processed = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for entry in data:
        # Normalize text
        text = entry.get("text", "").lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        title = entry.get("title", "").lower()
        title = re.sub(r'\s+', ' ', title)  # Remove extra spaces
        title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
        
        # Tokenize and remove stopwords
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        
        words_t = word_tokenize(title)
        words_t = [lemmatizer.lemmatize(word) for word in words_t if word not in stop_words]
        
        # Format publishing_date
        raw_date = entry.get("publishing_date", "")
        formatted_date = datetime.fromisoformat(raw_date).strftime('%Y-%m-%d') if raw_date else None

        #  Preprocess authors
        authors = entry.get("authors", [])  # Default to empty list
        if not isinstance(authors, list):
            authors = []  # Handle cases where authors might not be a list
        authors = sorted(set([author.strip().lower() for author in authors if author.strip()]))
        
        # Preprocess topics
        topics = entry.get("topics", [])  # Default to empty list
        if not isinstance(topics, list):
            topics = []  # Handle cases where topics might not be a list
        topics = sorted(set([topic.strip().lower() for topic in topics if topic.strip()]))

        # Collect processed data
        processed.append({
            "title":" ".join(words_t),
            "text": " ".join(words),
            "authors": authors,
            "topics": topics,# Unique topics
            "publishing_date": formatted_date
        })
    return processed

# Run preprocessing
with open('news_articles.json','r',encoding="utf-8") as file:
    data = json.load(file)
    preprocessed_data = preprocess_json(data)

output_path = 'preprocessed_news_articles.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, indent=4)
print("File processed successfully.")
