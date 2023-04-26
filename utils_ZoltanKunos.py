import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Define the function to preprocess and clean the text data
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    # Skip question words
    stop_words.difference_update({'when', 'which', 'why', 'how', 'what', 'where', 'who'})
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Combine tokens back into a single string
    clean_text = " ".join(tokens)

    return clean_text


def serialize_list_to_file(lst, filename):
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)


def deserialize_list_from_file(filename):
    with open(filename, 'rb') as file:
        deserialized_list = pickle.load(file)
    return deserialized_list