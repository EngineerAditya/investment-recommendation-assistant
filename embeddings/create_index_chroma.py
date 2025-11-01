import pandas as pd
import sqlite3
import os
import json
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. Configuration ---
# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.json')

# --- 2. Data Loading Functions ---

def load_csv_data(filepath, template):
    """Loads a CSV using a path and a template string."""
    documents = []
    # Make path absolute relative to project root
    if not os.path.isabs(filepath):
        filepath = os.path.join(PROJECT_ROOT, filepath)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Skipping.")
        return documents
        
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            try:
                # Apply the template using column names
                text = template.format(**row.to_dict())
                doc = Document(page_content=text, metadata={"source": filepath})
                documents.append(doc)
            except KeyError as e:
                print(f"  Warning: Skipping row. Missing key {e} in template for {filepath}")
        print(f"Successfully loaded {len(documents)} docs from {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}.")
    return documents

def load_sqlite_data(db_path, query, columns, template):
    """Loads from an SQLite DB using query, column names, and a template."""
    documents = []
    # Make path absolute relative to project root
    if not os.path.isabs(db_path):
        db_path = os.path.join(PROJECT_ROOT, db_path)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}. Skipping.")
        return documents
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        results = cursor.execute(query).fetchall()
        if not results:
            print(f"Warning: Query returned 0 results from {db_path}. Skipping.")
            conn.close()
            return documents
            
        for row in results:
            if len(row) != len(columns):
                print(f"  Error: Mismatch in {db_path}. Query returned {len(row)} items, but config lists {len(columns)} columns. Skipping row.")
                continue
            
            # Create a dictionary (e.g., {'product_name': 'Laptop', 'desc': '...'})
            row_dict = dict(zip(columns, row))
            
            try:
                # Apply the template
                text = template.format(**row_dict)
                doc = Document(page_content=text, metadata={"source": db_path, **row_dict})
                documents.append(doc)
            except KeyError as e:
                print(f"  Warning: Skipping row. Missing key {e} in template for {db_path}")

        conn.close()
        print(f"Successfully loaded {len(documents)} docs from {db_path}")
    except Exception as e:
        print(f"Error processing {db_path}: {e}.")
    return documents

# --- 3. Main Indexing Logic ---

def build_index():
    print("="*60)
    print("Investment Recommendation Assistant - Index Builder")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config File: {CONFIG_FILE}\n")
    
    # Load configuration
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found. Please create it.")
        return
        
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Get model name from config or use default
    model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
    
    print(f"Loading embedding model '{model_name}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return
    
    data_sources = config.get('data_sources', [])
    if not data_sources:
        print("Error: No 'data_sources' found in config.json.")
        return

    all_documents = []
    print("\n--- Starting Data Ingestion ---")
    for source in data_sources:
        source_type = source.get('type')
        path = source.get('path')
        template = source.get('template')

        if not all([source_type, path, template]):
            print(f"Warning: Skipping a source. Missing 'type', 'path', or 'template'.")
            continue

        if source_type == "csv":
            docs = load_csv_data(path, template)
            all_documents.extend(docs)
        elif source_type == "sqlite":
            query = source.get('query')
            columns = source.get('columns')
            if not all([query, columns]):
                print(f"Warning: Skipping sqlite source {path}. Missing 'query', 'columns'.")
                continue
            docs = load_sqlite_data(path, query, columns, template)
            all_documents.extend(docs)
        else:
            print(f"Warning: Unknown source type '{source_type}' for {path}. Skipping.")

    if not all_documents:
        print("\nNo documents were loaded. Exiting. Please check your config.json.")
        return

    print(f"\n--- Ingestion Complete ---")
    print(f"Total documents to index: {len(all_documents)}")

    # --- Create Chroma Vector Store ---
    print("Creating Chroma vector database...")
    
    # Get output directory from config or use default
    output_dir = config.get('output_directory', 'data/vector_db/chroma_db')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing database if it exists
    import shutil
    if os.path.exists(output_dir):
        print(f"Clearing existing database at {output_dir}...")
        shutil.rmtree(output_dir)
    
    # Create the Chroma vector store
    print("Generating embeddings and storing in Chroma...")
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=output_dir
    )
        
    print("\n--- Index Created Successfully ---")
    print(f"Chroma database created at: {output_dir}")
    print(f"Total documents indexed: {len(all_documents)}")

if __name__ == "__main__":
    build_index()

