import sqlite3
import os
import worldnewsapi
from worldnewsapi.rest import ApiException
from dotenv import load_dotenv  # <-- 1. Import the load_dotenv function

# --- CONFIGURATION ---
load_dotenv()  # <-- 2. Load the variables from your .env file

# <-- 3. Get the API key from environment variables
API_KEY = os.getenv("WORLDNEWSAPI_KEY") 

# Get the project root (parent directory of data-scrapers)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "news", "news.db")

def setup_database():
    """
    Creates the 'articles' table in our database.
    """
    print("Setting up database...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT NOT NULL,
            summary TEXT,
            url TEXT UNIQUE,
            source_name TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            published_at TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database setup complete.")


def fetch_and_store_news():
    """
    Fetches news using the new 'worldnewsapi' and stores
    it in our SQLite database.
    """
    print("Fetching latest news from worldnewsapi.com...")
    
    # <-- 4. Updated the check for a missing API key
    if not API_KEY:
        print("="*50)
        print("ERROR: WORLDNEWSAPI_KEY not found.")
        print("Please add it to your .env file.")
        print("="*50)
        return
    
    # 1. Initialize the API configuration (from docs)
    newsapi_configuration = worldnewsapi.Configuration(api_key={'apiKey': API_KEY})
    
    try:
        # 2. Instantiate the API object
        newsapi_instance = worldnewsapi.NewsApi(worldnewsapi.ApiClient(newsapi_configuration))

        # 3. Make the API Call
        response = newsapi_instance.search_news(
            text='(India OR NSE OR BSE OR RBI OR Sensex) AND (finance OR business OR stocks)',
            language='en',
            categories='business',
            sort="publish-time",
            sort_direction="DESC",
            number=100
        )

    except ApiException as e:
        print(f"Exception when calling NewsApi->search_news: {e}\n")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # 4. Get the list of articles from the response
    articles = response.news
    if not articles:
        print("No articles found. Check your API key or query parameters.")
        return

    print(f"Found {len(articles)} articles. Processing and storing...")

    # 5. Connect to our database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    articles_added = 0
    for article in articles:
        # 6. Map the API response to our database schema
        headline = article.title
        summary = article.text
        url = article.url
        pub_time = article.publish_date
        
        if article.authors and len(article.authors) > 0:
            source = ", ".join(article.authors)
        else:
            source = "Unknown"
        
        score = article.sentiment
        sentiment_label = "neutral"
        if score > 0.1:
            sentiment_label = "positive"
        elif score < -0.1:
            sentiment_label = "negative"
        
        if not url:
            continue
            
        # 7. Insert into database
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (headline, summary, url, source_name, sentiment, sentiment_score, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (headline, summary, url, source, sentiment_label, score, pub_time))
            
            if cursor.rowcount > 0:
                articles_added += 1
                
        except sqlite3.Error as e:
            print(f"Error inserting article: {e} - {url}")

    # 8. Save changes and close
    conn.commit()
    conn.close()
    
    print(f"Process complete. Added {articles_added} new articles to '{DB_PATH}'.")

# --- Main entry point ---
if __name__ == "__main__":
    setup_database()
    fetch_and_store_news()