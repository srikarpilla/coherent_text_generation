import streamlit as st
import cohere
import numpy as np
from numpy.linalg import norm
import os

# Initialize Cohere client with API key from environment variable
try:
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Cohere client: {str(e)}")
    st.stop()

# Streamlit layout
st.title("Srikar's AI Text Generation Model (Latest Cohere Chat API)")
st.write("Enter a prompt or question, and the app will generate a response using Cohere's Chat API.")

user_prompt = st.text_area("Enter your prompt or question:")

if st.button("Generate Text"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt or question.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Use the chat method with messages (new API style)
                response = co.chat(
                    model="command-xlarge-nightly",
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                generated_text = response.choices[0].message.content

                st.subheader("Generated Text")
                st.write(generated_text)

                # Compute semantic similarity between prompt and generated response
                embeddings = co.embed(texts=[user_prompt, generated_text]).embeddings
                similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
                st.metric("Similarity: Prompt vs Generated Text", f"{similarity:.2f}")

            except Exception as e:
                st.error(f"Error while generating text or embeddings: {str(e)}")
