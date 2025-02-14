import requests
import json

NEWS_API_KEY = "21f8cdead5c84a7283b01e08631a0107"

# Fetched 479 News Articles
class NewsFetcherAPI:
    def __init__(self, news_api_key):
        """
        Initialize the aggregator with a base URL, list of categories, and News API key.
        :param news_api_key: API key for News API.
        """
        self.articles = []
        self.news_api_key = news_api_key

    def fetch_from_news_api(self, queries, language='en', page_size=100):
        """
        Fetch news data using News API for multiple queries.
        :param queries: List of search queries for the API.
        :param language: Language of the articles (default: 'en').
        :param page_size: Number of articles to fetch per request.
        """
        try:
            url = "https://newsapi.org/v2/everything"

            for query in queries:
                params = {
                    'q': query,
                    'language': language,
                    'pageSize': page_size,
                    'apiKey': self.news_api_key
                }
                response = requests.get(url, params=params, verify=False)
                response.raise_for_status()
                data = response.json()

                for article in data.get('articles', []):
                    if article.get('title') == "[Removed]":
                        continue  
                    else:
                      self.articles.append({
                          'title': article.get('title', 'N/A'),
                          'text': article.get('content', 'N/A'),
                          'authors': article.get('author', 'N/A'),
                          'topics': query,
                          'publishing_date': article.get('publishedAt', 'N/A')
                      })

        except requests.RequestException as e:
            print(f"Error fetching from News API: {e}")

    def save_to_json(self, filename):
        """
        Save the aggregated news data to a JSON file.
        :param filename: Name of the output JSON file.
        """
        if not self.articles:
            print("No articles to save.")
            return

        with open(filename, 'a', encoding='utf-8') as json_file:
            if 'title' !="[Removed]":
              json.dump(self.articles, json_file, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")

# Step 2: Usage example
if __name__ == "__main__":
    aggregator = NewsFetcherAPI(NEWS_API_KEY)
    queries = ["Technology","News", "Sport", "Business", "Innovation", "Culture", "Arts", "Travel", "Earth"]
    aggregator.fetch_from_news_api(queries)
    aggregator.save_to_json('news_articles.json')
