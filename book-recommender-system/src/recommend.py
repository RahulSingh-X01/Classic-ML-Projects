import pickle
import numpy as np

popular_df = pickle.load(open("artifacts/popular.pkl",    "rb"))
pivot = pickle.load(open("artifacts/pivot.pkl",      "rb"))
similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))
books = pickle.load(open("artifacts/books.pkl", "rb"))

def recommend_book(book_name):
    if book_name not in pivot.index:
        return []
    
    # find index of movie
    index = np.where(pivot.index == book_name)[0][0]
    
    # Find similar movies based on cosine similarity
    similar_books = sorted(
        list(enumerate(similarity[index])), key=lambda x:x[1], reverse=True
        )[1:6]
    
    result = []
    for i in similar_books:
        title     = pivot.index[i[0]]
        book_info = books[books['Book-Title'] == title].iloc[0]
        result.append({
            'title'  : book_info['Book-Title'],
            'author' : book_info['Book-Author'],
            'image'  : book_info['Image-URL-M']
        })
        
    return result