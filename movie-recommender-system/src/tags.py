from nltk.stem import PorterStemmer
from preprocess import preprocess_data

# Initialize PorterStemmer
ps = PorterStemmer()

# Stemming function
def stemming(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# main function
def create_tags():
    # Get the preprocessed the data
    movies = preprocess_data()
    
    # Create tags column
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Keep only Id, title and tags columns
    movies = movies[['movie_id', 'title', 'tags']]
    
    # Convert tags from list to string
    movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))
    
    # Coverting tags to lowercase
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    
    # Apply stemming to tags column
    movies['tags'] = movies['tags'].apply(stemming)
    
    return movies