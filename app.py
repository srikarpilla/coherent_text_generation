import streamlit as st
import cohere
import numpy as np
import os

# --- SECURELY LOAD API KEY ---
# The COHERE_API_KEY is loaded securely from Streamlit's secrets management.
# You will set this up in the .streamlit/secrets.toml file.
try:
    cohere_api_key = st.secrets["COHERE_API_KEY"]
except KeyError:
    st.error("Cohere API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# Instantiate the Cohere client
co = cohere.Client(cohere_api_key)

# Define a variable for the embeddings model
embedding_model = "embed-english-v3.0"

# --- STREAMLIT APP LAYOUT ---
st.title("Cohere Text & Embedding Analysis")
st.write("This app demonstrates text generation and semantic similarity analysis using the Cohere API.")
st.markdown("---")


# --- GENERATE CREATIVE TEXTS AND DISPLAY ---
st.header("1. Generated Texts")

# Use st.spinner to show a loading state while generating text
with st.spinner("Generating creative texts..."):
    # Generate the first response: a poem
    response_1 = co.generate(
        prompt="write a poem about the sky being blue",
        max_tokens=150,
        temperature=1,
    )

    # Generate the second response: an overview of machine learning
    response_2 = co.generate(
        prompt="give me a basic overview of machine learning",
        max_tokens=200,
        temperature=0.5,
    )

    # Generate the third response: a Flask app script
    response_3 = co.generate(
        prompt="write a python script for a basic flask app",
        max_tokens=250,
        temperature=0,
    )

# Create a tuple of the generated texts from the responses
phrases = (
    response_1.generations[0].text,
    response_2.generations[0].text,
    response_3.generations[0].text
)

# Display the generated texts in expanders for better UI
with st.expander("Poem about the sky"):
    st.write(phrases[0])

with st.expander("Overview of Machine Learning"):
    st.write(phrases[1])

with st.expander("Basic Flask App Script"):
    st.write(phrases[2])

st.markdown("---")

# --- UTILITY FUNCTION FOR COSINE SIMILARITY ---
def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# --- PROCESS AND COMPARE TEXTS ---
st.header("2. Semantic Similarity Analysis")

with st.spinner("Embedding phrases and calculating similarity..."):
    # Convert phrases into vector embeddings using the Cohere embed API
    response = co.embed(
        model=embedding_model,
        input_type='search_document',
        texts=phrases
    )
    embeddings = response.embeddings

    # Calculate and print the similarity between pairs of embeddings
    similarity_score_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    similarity_score_2_3 = cosine_similarity(embeddings[1], embeddings[2])
    similarity_score_1_3 = cosine_similarity(embeddings[0], embeddings[2])

st.write("Semantic similarity scores between the generated texts:")
st.write(f'**Poem vs. Machine Learning:** {similarity_score_1_2:.4f}')
st.write(f'**Machine Learning vs. Flask App:** {similarity_score_2_3:.4f}')
st.write(f'**Poem vs. Flask App:** {similarity_score_1_3:.4f}')
