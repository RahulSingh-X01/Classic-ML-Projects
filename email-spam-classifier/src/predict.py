import pickle
from preprocess import data_transformation

def load_artifacts(model_path=r"email-spam-classifier\models\model.pkl", vectorizer_path=r"email-spam-classifier\models\vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf

def predict(text, model, tfidf):
    cleaned_text = data_transformation(text)

    vector = tfidf.transform([cleaned_text]).toarray()

    prediction = model.predict(vector)[0]

    return "Spam" if prediction == 1 else "Not Spam"

def main():
    model, tfidf = load_artifacts()
    print("Email Spam Classifier — type 'quit' to exit\n")

    while True:
        text = input("Enter email text: ")

        if text.lower() == "quit":
            break

        result = predict(text, model, tfidf)
        print(f"Prediction: {result}\n")

    
if __name__ == "__main__":
    main()
