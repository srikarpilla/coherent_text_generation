import streamlit as st
import cohere
import numpy as np
from numpy.linalg import norm
import os

# Initialize Cohere client with API key from environment variable
try:
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
except KeyError:
    st.error("Cohere API key not found. Please set the COHERE_API_KEY environment variable in Render.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Cohere client: {str(e)}")
    st.stop()

# Streamlit app layout
st.title("Coherent Text Generation")
st.write("This app demonstrates text generation and semantic similarity using Cohere's API.")

# Generate three text snippets
st.header("Generated Texts")
if st.button("Generate Texts"):
    with st.spinner("Generating texts..."):
        try:
            poem = co.generate(prompt="Write a poem about AI", num_generations=1, max_tokens=100).generations[0].text
            ml_overview = co.generate(prompt="Brief overview of machine learning", num_generations=1, max_tokens=100).generations[0].text
            python_script = co.generate(prompt="Basic Python script for hello world", num_generations=1, max_tokens=100).generations[0].text
            
            st.subheader("Poem")
            st.write(poem)
            st.subheader("Machine Learning Overview")
            st.write(ml_overview)
            st.subheader("Python Script")
            st.code(python_script, language="python")

            # Calculate semantic similarity
            st.header("Semantic Similarity Scores")
            texts = [poem, ml_overview, python_script]
            embeddings = co.embed(texts=texts).embeddings
            labels = ["Poem", "ML Overview", "Python Script"]
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
                    st.metric(f"Similarity: {labels[i]} vs {labels[j]}", f"{similarity:.2f}")
        except Exception as e:
            st.error(f"Error generating texts or embeddings: {str(e)}")
