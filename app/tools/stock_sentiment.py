from langchain.agents import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openbb import obb
import pandas as pd
from app.tools.utils import wrap_dataframe


def analyze_sentiment(text):
    if text is None:
        return 0.0

    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        if "compound" in sentiment:
            return sentiment["compound"]
        else:
            return 0.0
    except Exception:
        return 0.0


@tool
def get_news_sentiment(symbol: str) -> str:
    """Get News Sentiment for a Stock."""
    try:
        import requests
        import os
        
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return "\n<observation>\nAlpha Vantage API key not found in environment variables\n</observation>\n"
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data or not data["feed"]:
            return "\n<observation>\nNo news found for the given symbol\n</observation>\n"
        
        news_data = []
        for item in data["feed"]:
            news_data.append({
                'title': item.get('title', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'time_published': item.get('time_published', ''),
                'summary': item.get('summary', ''),
                # Alpha Vantage provides its own sentiment
                'av_sentiment': item.get('overall_sentiment_score', 0)
            })
        
        df = pd.DataFrame(news_data)
        # Add our own sentiment analysis
        df['sentiment_score'] = df['title'].apply(analyze_sentiment)
        
        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"