Cohere Streamlit App
This is a Streamlit web application that demonstrates the capabilities of the Cohere API for text generation and semantic similarity analysis.

The app performs the following tasks:

Generates Creative Text: It uses Cohere's generate endpoint to create a poem, a brief overview of machine learning, and a basic Python script.

Calculates Semantic Similarity: It converts the generated texts into vector embeddings using the embed endpoint.

Displays Similarity Scores: It calculates the cosine similarity between the embeddings of the three texts to show how semantically related they are.

This project is designed to be easily deployed to Streamlit Cloud.

How to Run Locally
Prerequisites
Python 3.7 or higher

A Cohere API key

1. Clone the repository
git clone <your-repository-url>
cd <your-repository-name>

2. Set up a virtual environment
It's a best practice to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS and Linux
source venv/bin/activate

3. Install dependencies
Install the required packages using the requirements.txt file.

pip install -r requirements.txt

4. Configure your API Key
For security, do not hardcode your API key. Create a .streamlit folder in your project's root directory and add a secrets.toml file inside it.

The folder structure should look like this:

.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
└── README.md

Add your Cohere API key to .streamlit/secrets.toml as follows:

COHERE_API_KEY = "your_cohere_api_key_here"

Note: Replace your_cohere_api_key_here with your actual key. This file should not be committed to your public repository.

5. Run the Streamlit app
With your virtual environment activated, run the app using the following command:

streamlit run app.py

Your app will open in your default web browser.

Deployment to Streamlit Cloud
This application is ready to be deployed to Streamlit Cloud. The app.py, requirements.txt, and .streamlit/secrets.toml files are all the components needed for a successful deployment. Streamlit Cloud will handle the secure management of your API key and the installation of your dependencies.

Important: Ensure your repository is public for a seamless Streamlit Cloud deployment.

License
This project is open-sourced under the MIT license.