# Import necessary libraries
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
import ast
import random

# Set the new header image
header_image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/1*QbBtk_xjwQWDW7aCrmGwfw.jpeg"
st.image(header_image_url, use_column_width=True)

# Load the recommendations.csv file
df = pd.read_csv("Dataset/recommendations.csv")

# Data preprocessing for collaborative filtering
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['userId', 'movie_id', 'user_rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train collaborative filtering model
model = SVD()
model.fit(trainset)

# Function to get recommended movies based on user and genre
def get_recommendations(user_id, genre=None, title=None, search_all=False):
    if search_all:
        # Get all unique movie IDs
        all_movie_ids = df['movie_id'].unique()

        # Exclude movies that the user has already rated
        user_movies = df[df['userId'] == user_id]['movie_id'].tolist()
        movie_ids_to_predict = list(set(all_movie_ids) - set(user_movies))

        # Predict ratings for the movies
        predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids_to_predict]

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Check if there are recommendations
        if not predictions:
            # If no recommendations, display random movies
            random_movies = df.sample(10)
            for _, movie_info in random_movies.iterrows():
                display_movie_info(movie_info)

        return predictions
    else:
        if genre:
            # Filter movies by genre
            genre_filter = df['genres_list'].apply(ast.literal_eval).apply(lambda x: genre in x)
            movies = df.loc[genre_filter]
        elif title:
            # Filter movies by title
            movies = df[df['title'].str.contains(title, case=False)]
        else:
            return None

        # Get recommended movies for the user
        user_movies = df[df['userId'] == user_id]['movie_id'].tolist()
        movie_ids_to_predict = list(set(movies['movie_id']) - set(user_movies))

        # Predict ratings for the movies
        predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids_to_predict]

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        return predictions

# Function to display movie information
def display_movie_info(movie_info):
    st.subheader("Movie Information")

    # Display movie poster with reduced size and placeholder for missing image
    poster_url = movie_info['hyperlinks']
    try:
        st.image(poster_url, caption="Movie Poster", use_column_width=True, output_format="JPEG", width=200)
    except Exception as e:
        st.warning("Image not available. Using placeholder.")
        st.image("https://media.officedepot.com/images/f_auto,q_auto,e_sharpen/products/951851/951851_o02_112520/951851",
                 caption="Placeholder", use_column_width=True, output_format="JPEG", width=200)

    # Display movie title
    st.write(f"**Title:** {movie_info['title']}")

    # Display tagline if not blank
    tagline = movie_info['tagline']
    if tagline:
        st.write(f"**Tagline:** {tagline}")

    # Display genres
    genres = ', '.join(ast.literal_eval(movie_info['genres_list']))
    st.write(f"**Genres:** {genres}")

    # Display movie description
    st.write(f"**Description:** {movie_info['overview']}")

    # Display movie ratings
    st.write(f"**Rating:** {movie_info['vote_average']}")

    # Display movie duration
    st.write(f"**Duration:** {movie_info['runtime']} minutes")

# Function to get alternative movies based on genre
def get_alternative_by_genre(genre):
    genre_filter = df['genres_list'].apply(ast.literal_eval).apply(lambda x: genre in x)
    alternative_movies = df[genre_filter].sample(10).drop_duplicates()
    return alternative_movies

# Streamlit app
st.subheader("Using Content Boosted Collaborative Filtering")

# Select userId using a dropdown
user_id = st.selectbox("Choose userId (1-10)", list(range(1, 11)))

# Select between Genre, Title, and All
selection = st.radio("Select search criteria:", ["Genre", "Title", "All"])

if selection == "Title":
    # Search by movie title
    st.subheader("Enter the name of the movie:")
    movie_title = st.text_input("Movie Title:")
    if st.button("Search"):
        # Display loading message
        with st.spinner("Finding recommendations..."):
            st.image("https://realpython.com/cdn-cgi/image/width=960,format=auto/https://files.realpython.com/media/Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.451abc4ecb9f.jpg",
                     caption="Recommendations in progress", use_column_width=True, output_format="JPEG")

            # Process search and display results
            st.write(f"Searching for movie: {movie_title}")
            recommendations = get_recommendations(user_id, title=movie_title)
            if recommendations:
                # Display the top recommendations
                seen_movies = set()
                for prediction in recommendations:
                    movie_info = df[df['movie_id'] == prediction.iid].iloc[0]
                    if movie_info['movie_id'] not in seen_movies:
                        seen_movies.add(movie_info['movie_id'])
                        display_movie_info(movie_info)
            else:
                st.warning("No movies found with the title. Try again with a different movie name.")

elif selection == "Genre":
    # Extract unique genres
    genres_list = df['genres_list'].apply(ast.literal_eval).explode().unique()
    selected_genre = st.selectbox("Choose genre:", genres_list)
    if st.button("Search"):
         # Display loading message
        with st.spinner("Finding recommendations..."):
            st.image("https://realpython.com/cdn-cgi/image/width=960,format=auto/https://files.realpython.com/media/Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.451abc4ecb9f.jpg",
                     caption="Recommendations in progress", use_column_width=True, output_format="JPEG")

            # Process search and display results
            st.write(f"Searching for movies in genre: {selected_genre}")
            recommendations = get_recommendations(user_id, genre=selected_genre)
            if recommendations:
                # Display the top recommendations
                seen_movies = set()
                for prediction in recommendations:
                    movie_info = df[df['movie_id'] == prediction.iid].iloc[0]
                    if movie_info['movie_id'] not in seen_movies:
                        seen_movies.add(movie_info['movie_id'])
                        display_movie_info(movie_info)

                # Display additional movies with the selected genre
                additional_movies = get_alternative_by_genre(selected_genre)
                if additional_movies is not None and not additional_movies.empty:
                    st.subheader("Additional Movies with the Selected Genre:")
                    seen_movies = set()
                    for _, movie_info in additional_movies.head(5).iterrows():
                        if movie_info['movie_id'] not in seen_movies:
                            seen_movies.add(movie_info['movie_id'])
                            display_movie_info(movie_info)
            else:
                # Display alternative movies based on genre
                alternative_movies = get_alternative_by_genre(selected_genre)
                if alternative_movies is not None and not alternative_movies.empty:
                    seen_movies = set()

                    for _, movie_info in alternative_movies.head(5).iterrows():
                        if movie_info['movie_id'] not in seen_movies:
                            seen_movies.add(movie_info['movie_id'])
                            display_movie_info(movie_info)
                else:
                    st.warning("No movies found with the selected genre.")

elif selection == "All":
    if st.button("Search All Movies"):
        # Display loading message
        with st.spinner("Finding recommendations..."):
            st.image("https://realpython.com/cdn-cgi/image/width=960,format=auto/https://files.realpython.com/media/Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.451abc4ecb9f.jpg",
                     caption="Recommendations in progress", use_column_width=True, output_format="JPEG")

            # Process search and display results
            st.write("Searching for all movies")
            recommendations = get_recommendations(user_id, search_all=True)
            if recommendations:
                # Display the top recommendations
                seen_movies = set()
                for prediction in recommendations:
                    movie_info = df[df['movie_id'] == prediction.iid].iloc[0]
                    if movie_info['movie_id'] not in seen_movies:
                        seen_movies.add(movie_info['movie_id'])
                        display_movie_info(movie_info)
            else:
                random_movies = df.sample(10)
                for _, movie_info in random_movies.iterrows():
                    display_movie_info(movie_info)
