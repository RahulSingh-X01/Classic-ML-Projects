from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tags import create_tags

cv = CountVectorizer(max_features=5000, stop_words='english')

def vectorize():
    # Get movies dataframe
    movies = create_tags()
    
    # Apply countervectorizer on tags
    vectors = cv.fit_transform(movies['tags']).toarray()
    
    # Calculate cosine similarity between vectors
    similarity = cosine_similarity(vectors)
    
    return similarity
    