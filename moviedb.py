import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import pickle

# Load the dataset
movies_data = pd.read_csv('movies.csv')

# Fill NaN values and combine selected features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

movies_data['combined_features'] = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

# Function to recommend movies
def recommend_movies(movie_name, num_recommendations=10):
    list_of_all_titles = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print("No similar movies found. Please check the spelling or try another movie.")
        return

    close_match = close_matches[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Get similarity scores for all movies
    similarity_scores = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies suggested for you based on '{close_match}':\n")
    count = 0
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        if count < num_recommendations:
            print(f"{count + 1}. {title_from_index}")
            count += 1
        else:
            break

# Save the model to a file
model_data = {
    'vectorizer': vectorizer,
    'similarity': similarity,
    'movies_data': movies_data,
    'selected_features': selected_features
}

with open('movie_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(model_data, model_file)

print("The model has been saved to 'movie_recommendation_model.pkl'.")



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
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

st.title("Movie Recommendation System")

movie_name = st.text_input("Enter your favorite movie name:")
num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name, num_recommendations)
    if recommendations:
        st.write(f"Movies suggested for you based on '{movie_name}':")
        for i, movie in enumerate(recommendations):
            st.write(f"{i + 1}. {movie}")
