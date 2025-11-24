# FinBuddy AI Investment Assistant 💰

An advanced AI-powered financial advisory system built with LangGraph that provides personalized investment recommendations for Indian markets. FinBuddy combines RAG (Retrieval-Augmented Generation), live market data, and LLM reasoning to create balanced portfolio strategies.

---

## 🚀 Project Overview

FinBuddy is a sophisticated financial advisory system that leverages:
- **RAG-based Market Analysis**: Vector database of market news for real-time sentiment analysis
- **Live Financial Data**: Real-time stock metrics via yfinance API
- **Multi-Tool Agent System**: LangGraph workflow with 3 specialized tools
- **Indian Market Expertise**: SEBI guidelines, tax optimization (80C, ELSS), ₹-based calculations
- **Dual Interface**: CLI (rag_app.py) and Web UI (streamlit_app.py)
- **Risk Profiling**: Age-based, goal-based portfolio construction
- **Calculation Accuracy**: Step-by-step formula verification

---

## ✨ Key Features

### 1. **RAG-Powered Market Analysis**
Vector database of market news articles with sentiment analysis for informed decision-making.

### 2. **Live Stock Data Integration**
Real-time financial metrics (PE ratio, market cap, 52-week range, sector info) via yfinance API.

### 3. **Multi-Tool Agent System**
LangGraph workflow with 3 specialized tools:
- `get_market_news(query)` - Search vector DB for relevant news
- `get_stock_data(ticker)` - Fetch live stock fundamentals
- `create_portfolio(amount, risk, horizon, tickers)` - Generate allocation table

### 4. **Indian Market Expertise**
- SEBI-compliant recommendations
- Tax optimization strategies (80C limits, LTCG/STCG rates)
- ₹-based calculations
- Support for NSE/BSE tickers (.NS suffix)

### 5. **Dual Interface**
- **CLI**: `rag_app.py` - Command-line interface for quick queries
- **Web UI**: `streamlit_app.py` - Modern web interface with chat history

### 6. **Risk Profiling**
Intelligent asset allocation based on:
- User age and financial goals
- Risk tolerance (Low/Medium/High)
- Investment horizon
- Market conditions

### 7. **Calculation Accuracy**
Step-by-step formula verification with transparent reasoning:
```
Portfolio Return = (65% × 11%) + (25% × 6.5%) + (10% × 7%) = 9.48%
```

---

## 🏗️ Technical Architecture

### Core Components

**LangGraph Agent**: State machine with tool-calling capabilities that orchestrates the entire workflow

**Vector Store**: ChromaDB with HuggingFace embeddings (all-MiniLM-L6-v2) for efficient similarity search

**LLM**: Ollama (llama3:8b) - Local, private inference without external API dependencies

**Data Sources**:
- **Vector DB**: Market news articles with sentiment scores
- **yfinance**: Live stock fundamentals (PE ratio, market cap, etc.)
- **System Knowledge**: Investment strategies, SEBI guidelines, tax laws

### Agent Workflow

```
User Query → LangGraph → [Market News Tool / Stock Data Tool / Portfolio Builder]
                      ↓
              LLM Reasoning (llama3:8b)
                      ↓
              Response with Citations
```

**State Machine Flow**:
```
START → Agent (LLM decides) → Should Continue?
                                   ├─ Yes → Tools → Agent (loop)
                                   └─ No → __END__ (final response)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph + LangChain |
| **LLM** | Ollama (ChatOllama wrapper) |
| **Embeddings** | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB (persistent storage) |
| **Finance Data** | yfinance (Yahoo Finance API) |
| **News Scraping** | worldnewsapi |
| **Web UI** | Streamlit with custom CSS |
| **Config** | JSON-based configuration |
| **Data Processing** | pandas + numpy |

---

## 📁 Project Structure

```
investment-recommendation-assistant/
├── rag_app.py                 # CLI application
├── streamlit_app.py           # Streamlit web interface
├── system_prompt.py           # Enhanced advisory prompts
├── data-scrapers/
│   └── fetch_news.py         # News scraping from WorldNewsAPI
├── embeddings/
│   ├── create_index_chroma.py # Vector DB creation script
│   └── config.json           # Embedding/DB configuration
├── data/
│   ├── news/
│   │   └── news.db           # SQLite database for scraped news
│   └── vector_db/
│       └── chroma_db/        # ChromaDB persistent storage
├── requirements.txt          # Python dependencies
├── run.sh                    # CLI launcher with setup
├── run_streamlit.sh          # Web UI launcher
└── README.md                 # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Ollama installed locally
- Internet connection for initial setup

### Step-by-Step Setup

#### 1. Clone Repository
```bash
git clone https://github.com/EngineerAditya/investment-recommendation-assistant.git
cd investment-recommendation-assistant
```

#### 2. Install Ollama (Required)
Visit [https://ollama.ai/download](https://ollama.ai/download) and install Ollama for your platform.

Then pull the required model:
```bash
ollama pull llama3:8b
```

Verify installation:
```bash
ollama list
# Should show llama3:8b in the list
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n finbuddy python=3.10
conda activate finbuddy
pip install -r requirements.txt
```

#### 4. Configure Environment
Create a `.env` file in the project root:
```bash
WORLDNEWSAPI_KEY=your_key_here  # Get from https://worldnewsapi.com
```

**Note**: Get your free API key from [WorldNewsAPI](https://worldnewsapi.com)

#### 5. Scrape Market News
```bash
python data-scrapers/fetch_news.py
```

This will:
- Fetch latest Indian market news
- Store in `data/news/news.db` (SQLite database)
- Filter by Indian financial terms (NSE, BSE, RBI, Sensex, etc.)

#### 6. Create Vector Database
```bash
python embeddings/create_index_chroma.py
```

This will:
- Load news from SQLite database
- Generate embeddings using HuggingFace model
- Store in ChromaDB at `data/vector_db/chroma_db/`

#### 7. Run Application

**Option A: CLI Interface**
```bash
python rag_app.py
# or
./run.sh
```

**Option B: Web Interface**
```bash
streamlit run streamlit_app.py
# or
./run_streamlit.sh
```

The web interface will open at `http://localhost:8501`

---

## 🔍 How It Works

### 1. Data Pipeline

```
WorldNewsAPI → fetch_news.py → data/news/news.db (SQLite)
                                       ↓
                          create_index_chroma.py
                                       ↓
                          Embeddings (all-MiniLM-L6-v2)
                                       ↓
                          ChromaDB (data/vector_db/chroma_db/)
```

### 2. Agent Tools (3 Tools)

#### **Tool 1: get_market_news(query)**
- **Purpose**: Search vector DB for relevant news articles
- **Returns**: Headlines, summaries, sentiment scores, URLs
- **Use Case**: "Current market outlook", "IT sector news", "banking sector"
- **Example**:
  ```python
  get_market_news("Nifty outlook indian markets")
  # Returns: Recent news with sentiment analysis
  ```

#### **Tool 2: get_stock_data(ticker)**
- **Purpose**: Fetch live stock data via yfinance
- **Returns**: Current price, PE ratio, market cap, 52-week range, sector
- **Use Case**: "TCS.NS", "RELIANCE.NS" (Indian stocks with .NS suffix)
- **Example**:
  ```python
  get_stock_data("TCS.NS")
  # Returns: Live financial metrics from Yahoo Finance
  ```

#### **Tool 3: create_portfolio(amount, risk, horizon, tickers)**
- **Purpose**: Generate asset allocation table
- **Risk-based Strategy**:
  - Low Risk: 40% cash/bonds, 60% equity
  - Medium Risk: 15% cash/bonds, 85% equity
  - High Risk: 100% equity
- **Returns**: Formatted portfolio with percentages and amounts
- **Example**:
  ```python
  create_portfolio(500000, "medium", 7, ["TCS.NS", "HDFCBANK.NS"])
  # Returns: Detailed allocation table
  ```

### 3. LangGraph State Machine

The agent uses a state machine to decide when to:
1. Call tools for more information
2. Process information and reason
3. Provide final response

**Flow**:
```
User: "I have ₹5 lakhs to invest"
  ↓
Agent analyzes query
  ↓
Calls get_market_news("market outlook")
  ↓
Calls get_stock_data("TCS.NS") [multiple times]
  ↓
Calls create_portfolio(500000, "medium", 7, [...])
  ↓
Synthesizes response with reasoning
```

### 4. Advisory Logic

From the enhanced system prompt (940+ lines):

**Market Context Protocol**: 
- ALWAYS call `get_market_news` first before recommendations
- Never hallucinate market data - use actual API calls

**Risk Profiling**:
- Age + goals + stability → Low/Medium/High risk profile

**Asset Allocation Guidelines**:
- **Low Risk**: 30-40% equity, 50-60% debt, 5-10% gold
- **Medium Risk**: 60-70% equity, 20-25% debt, 5-10% gold
- **High Risk**: 80-90% equity, 5-10% debt, 5% gold

**Stock Selection**:
- 5-7 diversified stocks across sectors
- All with ACTUAL data from get_stock_data tool
- Valuation analysis (PE ratio vs sector average)

**Calculation Verification**:
- Step-by-step formulas shown
- Totals must equal 100%
- All amounts cross-verified

**Tax Optimization**:
- 80C limits (₹1.5 lakh)
- ELSS recommendations
- LTCG/STCG tax rates

---

## ⚙️ Configuration

### embeddings/config.json
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "output_directory": "data/vector_db/chroma_db",
  "data_sources": [
    {
      "type": "sqlite",
      "path": "data/news/news.db",
      "query": "SELECT headline, summary, source_name, sentiment, published_at, url FROM articles",
      "columns": ["headline", "summary", "source_name", "sentiment", "published_at", "url"],
      "template": "News Headline: {headline}. Summary: {summary}. (Source: {source_name}, Sentiment: {sentiment}, Published: {published_at})"
    }
  ]
}
```

**Configurable Parameters**:
- `embedding_model`: HuggingFace model for embeddings
- `output_directory`: ChromaDB storage path
- `retriever_k`: Number of top results (default: 5)
- `ollama_model`: LLM model name (default: llama3:8b)

---

## 💬 Example Conversation Flow

```
User: I have ₹5 lakhs to invest, I'm 28 years old, medium risk tolerance, 7-year horizon

Agent Workflow:
1. Calls get_market_news("nifty outlook indian markets")
   → Returns: "Neutral sentiment, consolidating at 19,500"

2. Calls get_market_news("IT sector"), get_market_news("banking sector")
   → Identifies promising sectors with positive sentiment

3. Calls get_stock_data("TCS.NS")
   → Returns: PE 28, Market Cap ₹14L Cr, Price ₹3,800
   
4. Calls get_stock_data("INFY.NS"), get_stock_data("HDFCBANK.NS"), etc.
   → Fetches metrics for 7 stocks total

5. Analyzes: PE 28 vs sector avg 30 → undervalued

6. Calls create_portfolio(500000, "medium", 7, ["TCS.NS", "HDFCBANK.NS", ...])

7. Returns: Detailed allocation with reasoning + formulas
   Portfolio Breakdown:
   - Equity (65%): ₹3,25,000 across 7 stocks
   - Debt (25%): ₹1,25,000 in bonds/debt MFs
   - Gold (10%): ₹50,000 in Gold ETFs
   
   Expected Return Calculation:
   = (65% × 11%) + (25% × 6.5%) + (10% × 7%)
   = 7.15% + 1.625% + 0.7%
   = 9.48% annually
```

---

## 🎯 Key Differentiators

1. **Tool-First Architecture**: Never hallucinates data - all metrics from actual API calls
2. **Indian Market Focus**: .NS tickers, ₹ currency, 80C tax laws, SEBI awareness
3. **Calculation Transparency**: Shows formulas and step-by-step reasoning
4. **Conflicting Goals Handling**: Separates short-term (wedding fund) from long-term portfolios
5. **Market-Aware**: Adjusts lump sum vs SIP based on market valuations
6. **Privacy-First**: All LLM inference runs locally (no data sent to external APIs)
7. **Session Memory**: Maintains conversation context using MemorySaver checkpointer

---

## 🎨 Streamlit UI Features

From `streamlit_app.py`:
- **Custom CSS**: Professional design with gradient headers
- **Chat History**: Formatted markdown responses with proper styling
- **Tool Activity Expander**: Real-time view of agent tool calls
- **Sidebar**: Quick start guide and instructions
- **Clear Chat Button**: Reset conversation anytime
- **Mobile-Responsive**: Works on desktop and mobile browsers
- **Message Styling**:
  - User messages: Blue theme
  - Assistant messages: Green theme
  - Proper markdown rendering with tables and lists

---

## 📰 Data Scrapers

### data-scrapers/fetch_news.py

**Purpose**: Scrape latest Indian financial news

**Features**:
- Uses WorldNewsAPI for reliable news data
- Filters by keywords: India, NSE, BSE, RBI, Sensex
- Categories: Business and finance
- Stores in SQLite database with schema:
  - `headline`: Article title
  - `summary`: Article summary
  - `source_name`: News source
  - `sentiment`: Sentiment score
  - `published_at`: Publication timestamp
  - `url`: Article URL

**Usage**:
```bash
python data-scrapers/fetch_news.py
```

**Output**: `data/news/news.db` (SQLite database)

---

## 📦 Dependencies Breakdown

```
langchain + langgraph     # Agent framework for LLM orchestration
chromadb                  # Vector store for embeddings
sentence-transformers     # HuggingFace embeddings (all-MiniLM-L6-v2)
yfinance                  # Stock data from Yahoo Finance
ollama                    # Local LLM inference
streamlit                 # Web UI framework
worldnewsapi              # News scraping API
pandas + numpy            # Data processing and manipulation
python-dotenv             # Environment variable management
requests + tqdm           # HTTP requests and progress bars
```

### Installation
All dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting Guide

### 1. Ollama Connection Failed
**Symptoms**: "Connection refused" or "Ollama not running"

**Solutions**:
```bash
# Check if Ollama is installed
ollama list

# If not installed, visit https://ollama.ai/download

# Check if service is running
ollama serve

# Pull the required model
ollama pull llama3:8b
```

### 2. Vector Store Error
**Symptoms**: "ChromaDB not found" or "No collection found"

**Solutions**:
```bash
# Recreate vector database
python embeddings/create_index_chroma.py

# Check if database exists
ls -la data/vector_db/chroma_db/

# If missing, ensure news is scraped first
python data-scrapers/fetch_news.py
```

### 3. No Market News Found
**Symptoms**: "No relevant market news found in the database"

**Solutions**:
```bash
# Check if .env file exists with API key
cat .env

# Should contain:
# WORLDNEWSAPI_KEY=your_key_here

# Scrape fresh news
python data-scrapers/fetch_news.py

# Verify database has data
python -c "import sqlite3; conn = sqlite3.connect('data/news/news.db'); print(conn.execute('SELECT COUNT(*) FROM articles').fetchone())"
```

### 4. Stock Data Errors
**Symptoms**: "Unable to fetch stock data" or "Ticker not found"

**Solutions**:
- Verify internet connection (yfinance requires internet)
- Use .NS suffix for Indian stocks (TCS.NS not TCS)
- Check if ticker exists on NSE/BSE
- Example valid tickers: TCS.NS, INFY.NS, HDFCBANK.NS, RELIANCE.NS

### 5. Streamlit Port Already in Use
**Symptoms**: "Port 8501 is already in use"

**Solutions**:
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502

# Or kill the existing process
lsof -ti:8501 | xargs kill -9
```

---

## ⚡ Performance Optimization

- **HNSW Indexing**: ChromaDB uses HNSW (Hierarchical Navigable Small World) for fast similarity search
- **Top-K Retrieval**: Retriever limited to top-k=5 results (configurable in config.json)
- **Local LLM**: Ollama runs locally (no API rate limits or network latency)
- **Session-Based Memory**: MemorySaver checkpointer for efficient conversation history
- **Persistent Vector Store**: ChromaDB storage eliminates re-embedding on each run
- **Batch Processing**: News scraping and embedding creation done offline

---

## 🔒 Security & Privacy

- **Local LLM Inference**: All AI processing happens locally via Ollama (no data sent to external LLM APIs)
- **User Data Privacy**: No user queries or portfolio data transmitted to third parties
- **API Key Security**: `.env` file in `.gitignore` to prevent accidental commits
- **External API Usage**: Only yfinance (public stock data) and WorldNewsAPI (public news)
- **No User Tracking**: No analytics or tracking systems
- **Open Source**: Fully transparent codebase

---

## 🚀 Future Enhancements

- [ ] **Real-Time Data Feeds**: Add BSE/NSE real-time feeds for live prices
- [ ] **Backtesting Engine**: Historical performance testing for strategies
- [ ] **Multi-User Authentication**: Support for multiple user accounts
- [ ] **Portfolio Tracking Dashboard**: Track invested portfolios over time
- [ ] **Email/SMS Alerts**: Price alerts and portfolio notifications
- [ ] **PDF Report Generation**: Export recommendations as PDF reports
- [ ] **Advanced Charts**: Interactive price charts and technical indicators
- [ ] **Goal-Based Planning**: Dedicated modules for retirement, education planning
- [ ] **Robo-Advisory**: Automated rebalancing suggestions
- [ ] **Tax Calculator**: Integrated LTCG/STCG tax calculator

---

## 🎥 Demo for Recruiters

### 3-Minute Walkthrough

1. **Show Streamlit UI** (http://localhost:8501)
   - Modern, professional interface
   - Chat-based interaction

2. **Example Query**: 
   ```
   "I have ₹5 lakhs, age 28, medium risk tolerance, 7-year investment horizon"
   ```

3. **Watch Tool Calls in Real-Time**:
   - Expand "Tool Activity" section
   - See agent calling:
     - get_market_news() for sentiment
     - get_stock_data() for 7+ stocks
     - create_portfolio() for allocation

4. **Explain Architecture**:
   - RAG retrieval from ChromaDB
   - yfinance for live data
   - LLM reasoning with llama3:8b
   - Tool-first approach (no hallucinations)

5. **Highlight Differentiators**:
   - Calculation accuracy with formulas
   - Indian market expertise (₹, .NS tickers, 80C)
   - Privacy-first (local LLM)
   - Production-ready code quality

### Key Technical Points to Emphasize

✅ **LangGraph State Machine**: Production-grade agent framework  
✅ **RAG Implementation**: Vector search with ChromaDB + HuggingFace embeddings  
✅ **Tool Integration**: Clean abstractions with proper error handling  
✅ **Indian Market Expertise**: Deep domain knowledge implementation  
✅ **Calculation Verification**: Formula transparency and accuracy  
✅ **Dual Interface**: Both CLI and Web UI for different use cases  
✅ **Privacy & Security**: Local-first architecture  

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or feedback:
- GitHub Issues: [Create an issue](https://github.com/EngineerAditya/investment-recommendation-assistant/issues)
- GitHub: [@EngineerAditya](https://github.com/EngineerAditya)

---

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It is NOT financial advice. Always consult with a certified financial advisor before making investment decisions. Past performance does not guarantee future results. Investments in securities market are subject to market risks.

---

**Built with ❤️ using LangGraph, Ollama, and ChromaDB**