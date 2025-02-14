# Crawled 10,000 news articles
from crawler import NewsCrawler
from serializer import JsonSerializer
from config import max_articles

class NewsProcessor:
    def __init__(self, file_name, processes=5, max_articles=10000):
        self.file_name = file_name
        self.processes = processes
        self.max_articles = max_articles

    def process_and_serialize(self):
        # Crawl articles
        crawler = NewsCrawler(self.processes, self.max_articles)
        articles = crawler.crawl_articles()

        # Convert articles to dictionary format
        articles_dict = [article.to_dict() for article in articles]

        # Serialize to JSON
        JsonSerializer.serialize_to_json(articles_dict, self.file_name)
        return len(articles)

# Example usage
if __name__ == "__main__":
    news_processor = NewsProcessor('news_articles.json', processes=5, max_articles=max_articles)
    articles_count = news_processor.process_and_serialize()
    print(f"Number of articles crawled: {articles_count}")
