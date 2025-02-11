# Create environment
# pip install langchain wikipedia-api

# from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper

def initialize_wiki_api():
    wiki = WikipediaAPIWrapper()
    return wiki

def query_wikipedia(wiki, query, max_sentences=3):
    print(f"Querying Wikipedia with: {query}")
    results = wiki.run(query)
    limited_results = "".join(results.split(". ")[:max_sentences]) +"."
    return limited_results


if __name__ == "__main__":
    print("Starting Wikipedia Wrapper Demo")
    wiki = initialize_wiki_api()

    queries = [
        "Artificial intelligence",
        "Machine learning",
        "Langchain"
    ]

    for query in queries:
        result = query_wikipedia(wiki, query, max_sentences=3)
        print(f"\n> Result for '{query}':\n{result}")