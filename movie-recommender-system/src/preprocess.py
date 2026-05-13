import os
import ast
import pandas as pd


def convert(obj):
    names_list = []
    for i in ast.literal_eval(obj):
        names_list.append(i['name'])
    return names_list

def fetch_actors(obj):
    actors_list = []
    for i in ast.literal_eval(obj)[:4]:
        actors_list.append(i['name'])
    return actors_list

def fetch_director(obj):
    director = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            director.append(i['name'])
    return director

# Main preprocessig fucntion
def preprocess_data():
    # Load the datasets

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

    movies_data = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_movies.csv'))
    credits_data = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_credits.csv'))
    
    # Merge the datasets into one on the basis of title
    merged_data = movies_data.merge(credits_data, on='title')
    
    # Remove the useless columns 
    merged_data = merged_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    # Extract names from genres and keywords 
    merged_data['genres'] = merged_data['genres'].apply(convert)
    merged_data['keywords'] = merged_data['keywords'].apply(convert)
    
    # Fetch actors and director's name 
    merged_data['cast'] = merged_data['cast'].apply(fetch_actors)
    merged_data['crew'] = merged_data['crew'].apply(fetch_director)
    
    # Convert movie overview from string to list
    merged_data['overview'] = merged_data['overview'].apply(lambda x:x.split())
    
    # Strip all the spaces
    merged_data['genres'] = merged_data['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
    merged_data['keywords'] = merged_data['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
    merged_data['cast'] = merged_data['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
    merged_data['crew'] = merged_data['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
    
    return merged_data