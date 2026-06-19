import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    
    # Load the datasets
    books_data = pd.read_csv(os.path.join(DATA_DIR, 'Books.csv'))
    users_data = pd.read_csv(os.path.join(DATA_DIR, 'Users.csv'))
    ratings_data = pd.read_csv(os.path.join(DATA_DIR, 'Ratings.csv'))
    
    return books_data, users_data, ratings_data

#-------------------50 most popular books-----------------#
def popular_movies(top_n=50, min_ratings=250):
    
    books_data, _, ratings_data = load_data()
    # Merge ratings and books dataset on 'ISBN' column
    ratings_with_num = ratings_data.merge(books_data, on='ISBN')
    
    # Group data according to number of ratings
    num_rating_df = ratings_with_num.groupby('Book-Title').count()['Book-Rating'].reset_index()
    
    # Rename the book-rating column to num-rating
    num_rating_df = num_rating_df.rename(columns={'Book-Rating':'num-rating'})
    
    # Group data according to average rating per movie
    avg_rating_df = ratings_with_num.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    
    # Rename the book-rating column to avg-rating
    avg_rating_df = avg_rating_df.rename(columns={'Book-Rating':'avg-rating'})
    
    # Merge both the dataset with number of ratings and average rating
    popular_movies = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Select movies with more than 250 ratings
    popular_movies = popular_movies[popular_movies['num-rating'] > min_ratings]
    
    # Sort movies according to average rating in non-increasing order
    popular_movies = popular_movies.sort_values('avg-rating', ascending=False)
    
    # Select top 50 movies
    popular_movies = popular_movies.head(top_n)
    
    # merge with books_data
    popular_movies = popular_movies.merge(books_data, on='Book-Title')
    
    # Drop duplicate data rows
    popular_movies = popular_movies.drop_duplicates('Book-Title')
    
    # Select relevant columns
    popular_movies = popular_movies[['Book-Title', 'Book-Author', 'Image-URL-M', 'num-rating', 'avg-rating']]
    
    return popular_movies

def collaborative_model(min_book_rating=50, min_user_rating=200):
    books_data, user_data, ratings_data = load_data()
    
    # Merge ratings and books dataset on 'ISBN' column
    ratings_with_num = ratings_data.merge(books_data, on='ISBN')
    
    # Getting users who have given minimum 200 ratings
    users_with_high_ratings = ratings_with_num.groupby('User-ID').count()['Book-Rating'] > min_user_rating
    x = users_with_high_ratings[users_with_high_ratings].index
    
    # Getting books with minimum 50 ratings
    books_with_high_ratings = ratings_with_num.groupby('Book-Title').count()['Book-Rating'] > min_book_rating
    y = books_with_high_ratings[books_with_high_ratings].index
    
    # Filter data according to selected users and books
    filtered_rating = ratings_with_num[ratings_with_num['User-ID'].isin(x)]
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(y)]
    
    # Make a new dataframe with book-titles as indices
    movie_data = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
    
    # Get similarity score of each movie with every other movie accoridng to cosine similarity
    similarity_score = cosine_similarity(movie_data)
    
    return movie_data, similarity_score 

#--------------Save artifacts as pickel files------------#
def save_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    popular_df = popular_movies()
    pivot, similarity_score = collaborative_model()
    books_data, _, _ = load_data()

    pickle.dump(popular_df,       open(os.path.join(ARTIFACTS_DIR, "popular.pkl"), "wb"))
    pickle.dump(pivot,            open(os.path.join(ARTIFACTS_DIR, "pivot.pkl"), "wb"))
    pickle.dump(similarity_score, open(os.path.join(ARTIFACTS_DIR, "similarity.pkl"), "wb"))
    pickle.dump(books_data,       open(os.path.join(ARTIFACTS_DIR, "books.pkl"), "wb"))
    
#-----------------Main function-------------#
if __name__ == "__main__":
    save_artifacts()
    print("✅ All artifacts saved successfully!")
