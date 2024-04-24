import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
import os

# Load the OpenAI API key from environment variable
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
assert API_KEY, "ERROR: OpenAI Key is missing"

# Initialize the OpenAI client
client = openai.OpenAI(api_key=API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    """ Get the embedding for a given text using specified model """
    text = str(text).replace("\n", " ")  # Ensure text is a single line string
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def load_data(filename):
    """ Load the earnings CSV, rename the first column to 'text', and process embeddings """
    df = pd.read_csv(filename)
    first_column_name = df.columns[0]
    df.rename(columns={first_column_name: 'text'}, inplace=True)
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x))
    df['index_from_doc'] = df.index  # Add row index as a new column to track location in the original document
    return df

def main():
    # CSS to inject custom styles
    st.markdown("""
        <style>
        .stApp {
            background-color: #00171f;
        }
        .result-container {
            margin: 10px 0px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
        </style>
        """, unsafe_allow_html=True)

    # Streamlit application layout
    st.title('Earnings Call Search')

    # Step 1: Get the filename from user input
    filename = st.text_input("Enter the CSV filename:", 'msftq2.csv')

    # Initialize session state for earnings_df
    if 'earnings_df' not in st.session_state:
        st.session_state['earnings_df'] = pd.DataFrame()

    # Step 2: Load the data only after pressing the 'Load File' button
    if st.button('Load File', key='load_file'):
        st.session_state['earnings_df'] = load_data(filename)
        st.success("File loaded successfully!")

    # Step 3: Search term entry and process search only after pressing the 'Search' button
    search_term = st.text_input("Enter a search term:")

    if st.button('Search', key='search'):
        if search_term and not st.session_state['earnings_df'].empty:
            search_vector = get_embedding(search_term)
            # Calculate cosine similarity
            st.session_state['earnings_df']['similarity'] = st.session_state['earnings_df'].apply(
                lambda row: cosine_similarity([row['embedding']], [search_vector])[0][0], axis=1)
            st.session_state['earnings_df'].sort_values(by='similarity', ascending=False, inplace=True)

            # Display the results
            st.write("Top 10 Relevant Sentences from Earnings Call:")
            for index, row in st.session_state['earnings_df'].head(10).iterrows():
                color = "green" if row['similarity'] > 0.8 else "orange" if 0.7 < row['similarity'] <= 0.8 else "red"
                st.markdown(f"<div class='result-container' style='border-color:{color};'><strong>Rank {index+1}:</strong> {row['text']} <br><strong>Similarity:</strong> {row['similarity']:.2f}, <strong>Location in document:</strong> {row['index_from_doc']}</div>", unsafe_allow_html=True)
        else:
            st.error("Please load a file and enter a search term to start the search.")

if __name__ == "__main__":
    main()
