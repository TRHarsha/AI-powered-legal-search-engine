import streamlit as st
import PyMuPDF
from transformers import pipeline

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    text = ""
    with pymupdf.open(pdf_file) as pdf:
    for page_num in range(pdf.page_count):
        page = pdf[page_num]
        text += page.get_text("text")
    return text

def summarize_text(text):
    """Summarize the extracted text using Hugging Face pipeline."""
    # Summarization models often have token limits, so chunk the text
    max_chunk_size = 512
    text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    # Summarize each chunk and combine results
    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=100, min_length=25, do_sample=False)
        summarized_text += summary[0]["summary_text"] + " "
    return summarized_text

# Streamlit app layout
st.title("PDF Summarizer")
st.write("Upload a PDF file, and this tool will summarize its contents.")

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)
        st.write("### Extracted Text")
        st.write(extracted_text[:1000] + "...")  # Display the first 1000 characters for reference

    if st.button("Summarize Text"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(extracted_text)
            st.write("### Summary")
            st.write(summary)
