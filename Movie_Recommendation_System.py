import streamlit as st
import pandas as pd
import difflib
import pickle

# Load the model from the file
with open('movie_recommendation_model.pkl', 'rb') as model_file:
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

# Streamlit app
st.set_page_config(
    page_title="MoviesDB",
    page_icon="🍿",
)
st.title("Movie Recommendation System")

movie_name = st.text_input("Enter your favorite movie name:")
num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name, num_recommendations)
    if recommendations:
        st.write(f"Movies suggested for you based on '{movie_name}':")
        for i, movie in enumerate(recommendations):
            st.write(f"{i + 1}. {movie}")