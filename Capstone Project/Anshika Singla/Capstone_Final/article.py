# article.py
class ArticleSection:
    def __init__(self, headline, paragraphs):
        self.headline = headline
        self.paragraphs = paragraphs

    def to_dict(self):
        return {
            "headline": self.headline,
            "paragraphs": self.paragraphs
        }

class ArticleBody:
    def __init__(self, summary, sections):
        self.summary = summary
        self.sections = sections

    def to_dict(self):
        return {
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections]
        }

class Article:
    def __init__(self, title, text, authors, topics, publishing_date):
        self.title = title
        self.text = text
        self.authors = authors
        self.topics = topics
        self.publishing_date = publishing_date

    def to_dict(self):
        return {
            "title": self.title,
            "text": self.text.to_dict() if isinstance(self.text, ArticleBody) else self.text,
            "authors": self.authors,
            "topics": self.topics,
            "publishing_date": self.publishing_date.isoformat() 
        }
