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
st.title("Srikar's AI Text Generation Model")
st.write("Ask a question or give a prompt, and this app will generate text using Cohere's API.")

# User input for prompt
user_prompt = st.text_area("Enter your prompt or question here:")

if st.button("Generate Text"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt or a question to generate text.")
    else:
        with st.spinner("Generating text..."):
            try:
                # Generate text from user prompt
                generated_text = co.generate(
                    prompt=user_prompt,
                    num_generations=1,
                    max_tokens=150
                ).generations[0].text

                st.subheader("Generated Text")
                st.write(generated_text)

                # Semantic similarity: compare generated text with prompt embedding (optional enhancement)
                st.header("Semantic Similarity Scores (Optional)")
                embeddings = co.embed(texts=[user_prompt, generated_text]).embeddings
                similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
                st.metric(f"Similarity: Prompt vs Generated Text", f"{similarity:.2f}")

            except Exception as e:
                st.error(f"Error generating text or embeddings: {str(e)}")
