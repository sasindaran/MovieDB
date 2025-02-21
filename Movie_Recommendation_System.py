import streamlit as st
import pandas as pd
import difflib
import pickle
import os
import gdown  # Ensure gdown is in your requirements.txt

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="MoviesDB",
    page_icon="🍿",
)

# Define the filename and Google Drive file ID
pkl_file = 'movie_recommendation_model.pkl'
file_id = '1bF9UYcC4SviDwUV7vi573cihjZrGGiR3'
download_url = f'https://drive.google.com/uc?id={file_id}'

# Download the file if it doesn't exist locally
if not os.path.exists(pkl_file):
    st.write("Downloading the model file, please wait...")
    gdown.download(download_url, pkl_file, quiet=False)

# Load the model from the file
with open(pkl_file, 'rb') as model_file:
    model_data = pickle.load(model_file)

vectorizer = model_data['vectorizer']
similarity = model_data['similarity']
movies_data = model_data['movies_data']
selected_features = model_data['selected_features']

# Function to recommend movies
def recommend_movies(movie_name, num_recommendations=10):
    list_of_all_titles = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        st.write("No similar movies found. Please check the spelling or try another movie.")
        return []

    close_match = close_matches[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

    # Get similarity scores for all movies
    similarity_scores = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    count = 0
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        if count < num_recommendations:
            recommended_movies.append(title_from_index)
            count += 1
        else:
            break

    return recommended_movies

# Streamlit app layout
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter your favorite movie name:")
num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name, num_recommendations)
    if recommendations:
        st.write(f"Movies suggested for you based on '{movie_name}':")
        for i, movie in enumerate(recommendations):
            st.write(f"{i + 1}. {movie}")
