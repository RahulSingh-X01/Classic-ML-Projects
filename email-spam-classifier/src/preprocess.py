import nltk
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))
PUNCT = set(string.punctuation)


def load_data():
    data = pd.read_csv(
    r"C:\Users\rahul\Documents\Programming\Github Projects\Classic-ML-Projects\email-spam-classifier\data\spam.csv",
    encoding="latin-1"
    )
    
    return data


def data_transformation(text):
    ps = PorterStemmer()
    
    tokens = nltk.word_tokenize(text.lower())
    
    tokens = [t for t in tokens if t.isalnum()]
    
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in PUNCT]
    
    tokens = [ps.stem(t) for t in tokens]
    
    return " ".join(tokens)


def preprocess_data():
    data = load_data()
    
    data['v2'] = data[['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].fillna('').agg(' '.join, axis=1)
    data['v2'] = data['v2'].str.strip()
    
    data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    
    data.rename(columns={'v1':'label'}, inplace=True)
    data.rename(columns={'v2':'text'}, inplace=True)
    
    data = data.drop_duplicates(keep='first')
    
    data['label'] = data['label'].apply(lambda x: 1 if x=='spam' else 0)
    
    data['transformed_text'] = data['text'].apply(data_transformation)
    
    return data