import os
import pandas as pd

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
    
    