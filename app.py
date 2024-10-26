import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model for semantic search (Sentence-BERT for efficient embeddings)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Compact, fast model for embeddings

model = load_model()

# Load CSV and create embeddings for the entire data set
@st.cache_data
def load_data_and_embeddings():
    data = pd.read_csv(r"TRHarsha/AI-powered-legal-search-engine/blob/main/leg_dataset.csv")
    data['content'] = data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    embeddings = model.encode(data['content'].tolist(), show_progress_bar=True)
    return data, embeddings

data, embeddings = load_data_and_embeddings()

# Efficient Semantic Search using Embeddings
def search_with_embeddings(query, embeddings, data, top_k=5):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = scores.argsort()[-top_k:][::-1]  # Get indices of top k results
    return data.iloc[top_k_indices], scores[top_k_indices]

# Streamlit Interface
st.title("AI-Powered CSV Search Engine")

query = st.text_input("Enter your query:")
top_k = st.slider("Number of results:", 1, 20, 5)

if query:
    results, scores = search_with_embeddings(query, embeddings, data, top_k=top_k)
    for i, (index, row) in enumerate(results.iterrows()):
        st.write(f"**Result {i+1} (Score: {scores[i]:.2f})**")
        st.write(row['content'])
else:
    st.write("Enter a query to search the database.")
