#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Investment Recommendation Assistant"
echo "========================================"
echo ""

# Source conda properly
echo "--- Sourcing Conda ---"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

echo "--- Activating Conda Environment 'finbuddy' ---"
conda activate finbuddy

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'finbuddy'"
    echo "Please create it first: conda create -n finbuddy python=3.10"
    exit 1
fi

echo "--- STEP 1: Installing/Updating Requirements ---"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

# Check if fetch_news.py exists
if [ ! -f "data-scrapers/fetch_news.py" ]; then
    echo "Warning: 'data-scrapers/fetch_news.py' not found. Skipping news fetch."
else
    echo "--- STEP 2: Fetching Latest News ---"
    python data-scrapers/fetch_news.py
    
    if [ $? -ne 0 ]; then
        echo "Warning: News fetch failed, continuing anyway..."
    fi
fi

echo "--- STEP 3: Creating Vector Database ---"
python embeddings/create_index_chroma.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to create vector database"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "--- STEP 4: Starting RAG Application ---"
echo ""

python rag_app.py