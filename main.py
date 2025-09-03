import streamlit as st
import cohere
import os

# --- SECURELY LOAD API KEY ---
try:
    cohere_api_key = st.secrets["COHERE_API_KEY"]
except KeyError:
    st.error("Cohere API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# Instantiate the Cohere client
co = cohere.Client(cohere_api_key)

st.title(" Text Generation")
st.write("Enter a prompt and ")

# --- USER INPUT ---
prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate Text") and prompt.strip():
    with st.spinner("Generating text..."):
        response = co.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
        )
        generated_text = response.generations[0].text.strip()
    st.header("Generated Text")
    st.write(generated_text)
elif not prompt.strip():
    st.info("Type a prompt and click 'Generate Text' to begin.")

