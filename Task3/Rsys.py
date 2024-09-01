import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Step 2: Data Preprocessing
# Create a user-item matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN values with 0 (for simplicity, though other methods can be used)
user_movie_matrix = user_movie_matrix.fillna(0)

# Step 3: Calculate Similarity
# Compute the cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert the similarity matrix to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Step 4: Generate Recommendations
def get_recommendations(user_id, num_recommendations=5):
    # Get the similarity scores for the user
    similarity_scores = user_similarity_df[user_id]

    # Sort the similarity scores in descending order
    similar_users = similarity_scores.sort_values(ascending=False)

    # Get the top similar users (excluding the user itself)
    top_similar_users = similar_users.index[1:num_recommendations+1]

    # Get the movies watched by the similar users
    similar_users_movies = user_movie_matrix.loc[top_similar_users]

    # Compute the average ratings for each movie by the similar users
    average_ratings = similar_users_movies.mean(axis=0)

    # Sort the average ratings in descending order
    recommended_movies = average_ratings.sort_values(ascending=False).index

    # Get the movie titles
    recommended_movie_titles = movies[movies['movieId'].isin(recommended_movies)]['title']

    return recommended_movie_titles.head(num_recommendations)

# Example usage
user_id = 1  # Assuming we want recommendations for user with ID 1
num_recommendations = 5
recommendations = get_recommendations(user_id, num_recommendations)

print(f"Top {num_recommendations} movie recommendations for user {user_id}:")
for movie in recommendations:
    print(movie)
