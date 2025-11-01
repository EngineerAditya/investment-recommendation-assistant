# FinBuddy Streamlit Web Interface

A modern, user-friendly web interface for the FinBuddy AI Investment Assistant.

## 🌟 Features

- **💬 Interactive Chat Interface**: Conversational AI assistant for investment advice
- **📊 Real-time Market Data**: Live stock prices and financial metrics
- **📰 News Analysis**: Market sentiment from recent news articles
- **💼 Portfolio Creation**: Automated portfolio generation based on your risk profile
- **🇮🇳 Indian Market Focus**: Specialized in Indian stocks, mutual funds, and tax-saving instruments
- **🎨 Modern UI**: Clean, responsive design with real-time updates

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit
# or
pip install -r requirements.txt
```

### 2. Run the App

**Option A: Using the run script**
```bash
./run_streamlit.sh
```

**Option B: Direct command**
```bash
streamlit run streamlit_app.py
```

### 3. Access the App

Open your browser and navigate to:
```
http://localhost:8501
```

## 📋 How to Use

1. **Start a Conversation**: Type your investment query in the chat input at the bottom
2. **Provide Information**: Share details about:
   - Investment amount
   - Risk tolerance (low/medium/high)
   - Time horizon
   - Financial goals
   - Age and income (optional)

3. **Get Recommendations**: FinBuddy will:
   - Analyze your risk profile
   - Research market conditions
   - Suggest a diversified portfolio
   - Explain the rationale

4. **View Tool Activity**: Click the "Tool Activity" expander to see:
   - Market news searches
   - Stock data fetches
   - Portfolio calculations

## 🎯 Example Queries

```
I want to invest ₹5 lakhs with medium risk for 5 years
```

```
I'm 30 years old, earning ₹15 LPA, want to invest ₹10,000 monthly via SIP for retirement
```

```
How should I allocate ₹2 lakhs for my 8-year-old's education fund?
```

```
What are the best tax-saving investments under 80C?
```

## 🛠️ Features in Detail

### Chat Interface
- **Persistent History**: Your conversation is maintained throughout the session
- **Clear Chat**: Reset the conversation anytime using the sidebar button
- **Tool Transparency**: See what research FinBuddy is performing

### Sidebar Guide
- Quick start instructions
- Feature overview
- Session management
- Important disclaimers

### Responsive Design
- Works on desktop, tablet, and mobile
- Modern, professional appearance
- Color-coded messages (User vs Assistant)

## ⚙️ Configuration

The app uses the same configuration as the CLI version:

- **Config File**: `embeddings/config.json`
- **Vector Database**: `data/vector_db/chroma_db`
- **LLM Model**: Configured Ollama model (default: llama3:8b)

## 🔧 Customization

### Change Port
```bash
streamlit run streamlit_app.py --server.port 8080
```

### Enable External Access
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Custom Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 🐛 Troubleshooting

### "Ollama connection failed"
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`
- Install if needed: `ollama pull llama3:8b`

### "Vector store error"
- Run the indexing script first:
  ```bash
  python embeddings/create_index_chroma.py
  ```

### Port Already in Use
- Change the port:
  ```bash
  streamlit run streamlit_app.py --server.port 8502
  ```

## 📊 Comparison: CLI vs Streamlit

| Feature | CLI (`rag_app.py`) | Streamlit (`streamlit_app.py`) |
|---------|-------------------|--------------------------------|
| Interface | Terminal | Web Browser |
| Accessibility | Command line only | Any device with browser |
| History | Session-based | Persistent in UI |
| Tool Visibility | Console output | Expandable sections |
| User Experience | Technical | User-friendly |
| Sharing | Not suitable | Easy to share/demo |

## 🔐 Security Notes

- The app runs locally by default (localhost:8501)
- No data is sent to external servers (except yfinance API calls)
- Vector database and chat history stay on your machine
- For production deployment, consider authentication

## 📝 License

Same as the main FinBuddy project (see LICENSE file)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add authentication
- [ ] Save conversation history to database
- [ ] Export portfolio recommendations as PDF
- [ ] Add data visualization (charts, graphs)
- [ ] Multi-language support
- [ ] Voice input/output

## ⚠️ Disclaimer

This is an educational tool for learning about investments. It does NOT constitute financial advice. Always consult with SEBI-registered financial advisors before making investment decisions.
