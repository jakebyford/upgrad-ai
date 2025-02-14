# crawler.py
from fundus import CCNewsCrawler
from article import Article

class NewsCrawler:
    def __init__(self, *publishers, processes=5, max_articles=10000):
        self.crawler = CCNewsCrawler(*publishers, processes=processes)
        self.max_articles = max_articles

    def crawl_articles(self):
        articles = []
        for article in self.crawler.crawl(max_articles=self.max_articles):
            news_article = Article(
                title=article.title,
                text=article.body,  
                authors=article.authors,
                topics=article.topics,
                publishing_date=article.publishing_date
            )
            articles.append(news_article)
        return articles
