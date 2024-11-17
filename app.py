import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

# Load the model for semantic search (Sentence-BERT for efficient embeddings)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Compact, fast model for embeddings

model = load_model()

# Load CSV and create embeddings for the entire data set
@st.cache_data
def load_data_and_embeddings():
    data = pd.read_csv("leg_dataset.csv")
    data['content'] = data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    embeddings = model.encode(data['content'].tolist(), show_progress_bar=True)
    return data, embeddings

data, embeddings = load_data_and_embeddings()

# Efficient Semantic Search using Embeddings
def search_with_embeddings(query, embeddings, data, top_k=5):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = scores.argsort()[-top_k:][::-1]  # Get indices of top k results
    return data.iloc[top_k_indices], scores[top_k_indices], top_k_indices

# Calculate Precision and Recall
def calculate_precision_recall(top_k_indices, ground_truth_indices):
    y_true = [1 if idx in ground_truth_indices else 0 for idx in top_k_indices]
    y_pred = [1] * len(top_k_indices)  # Since all retrieved results are considered for evaluation
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall

# Calculate Mean Average Precision (MAP)
def calculate_map(top_k_indices, ground_truth_indices, k):
    aps = []
    for query_indices in top_k_indices:
        ap = 0
        relevant_count = 0
        for i, doc_idx in enumerate(query_indices):
            if doc_idx in ground_truth_indices:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        if relevant_count > 0:
            aps.append(ap / relevant_count)
    map_score = np.mean(aps)
    return map_score

# Streamlit Interface
st.title("AI-Powered Legal Search Engine")

query = st.text_input("Enter your query:")
top_k = st.slider("Number of results:", 1, 20, 5)

# Example ground truth for the query (manually labeled relevant document indices)
# In a real case, this should come from your labeled dataset or user feedback.
ground_truth_indices = [3, 5, 8]  # Example: indices of relevant documents for the query

if query:
    results, scores, top_k_indices = search_with_embeddings(query, embeddings, data, top_k=top_k)
    precision, recall = calculate_precision_recall(top_k_indices, ground_truth_indices)
    map_score = calculate_map([top_k_indices], [ground_truth_indices], top_k)
    
    # Display the results
    for i, (index, row) in enumerate(results.iterrows()):
        st.write(f"**Result {i+1} (Score: {scores[i]:.2f})**")
        st.write(row['content'])
    
    # Display precision, recall, and MAP metrics
    st.subheader("Test Results")

    # Precision, Recall, and MAP using st.metric
    st.metric(label="Precision", value=f"{precision:.2f}")
    st.metric(label="Recall", value=f"{recall:.2f}")
    st.metric(label="Mean Average Precision (MAP)", value=f"{map_score:.2f}")
    
    # Visualize the scores using bar chart
    st.subheader("Performance Visualization")
    metrics = ["Precision", "Recall", "MAP"]
    scores = [precision, recall, map_score]
    
    # Bar chart for visualizing the results
    fig, ax = plt.subplots()
    ax.bar(metrics, scores, color=['blue', 'green', 'orange'])
    ax.set_ylim([0, 1])  # Set y-axis limit from 0 to 1
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Score")
    
    st.pyplot(fig)
else:
    st.write("Enter a query to search the database.")
