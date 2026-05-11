from vectorize import vectorize

def recommend_movie(movie):
    # Get the movies and similarity scores
    movies, similarity = vectorize()
    
    # Fetch the movie index
    movie_index = movies[movies['title'] == movie].index[0]
    
    # Get the similarity array for movie
    distances = similarity[movie_index]
    
    # Calculate 5 similar movies
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    # Return the movies list
    recommendations = []
    for i in  movie_list:
        recommendations.append(movies.iloc[i[0]].title)
        
    return recommendations