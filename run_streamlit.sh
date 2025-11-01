#!/bin/bash
# Run FinBuddy Streamlit App

echo "🚀 Starting FinBuddy Web Interface..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Run the Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address localhost
