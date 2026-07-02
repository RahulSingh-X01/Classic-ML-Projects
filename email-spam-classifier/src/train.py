import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from preprocess import preprocess_data

def main():
    data = preprocess_data()
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['transformed_text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )

    tfidf = TfidfVectorizer()

    X_train = tfidf.fit_transform(X_train_text).toarray()

    X_test = tfidf.transform(X_test_text).toarray()

    model = LinearSVC()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Model and vectorizer saved to 'models/' folder.")

if __name__ == "__main__":
    main()