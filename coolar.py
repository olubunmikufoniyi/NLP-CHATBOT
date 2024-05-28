import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text file and preprocess the data
file_path = 'chatbot.txt'
encodings = ['utf-8', 'latin-1', 'utf-16']

data = None
for encoding in encodings:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = f.read().replace('\n', ' ')
        break  # If reading is successful, break the loop
    except UnicodeDecodeError:
        continue

if data is None:
    st.error("Failed to read the file with the available encodings.")
    st.stop()

# Tokenize the text into sentences
sentences = sent_tokenize(data)
original_sentences = sentences.copy()  # Keep a copy of the original sentences

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [' '.join(preprocess(sentence)) for sentence in sentences]

# Initialize the TF-IDF vectorizer and fit it on the corpus
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Define a function to find the most relevant sentences given a query
def get_most_relevant_sentences(query, top_n=3):
    # Preprocess the query
    query_processed = ' '.join(preprocess(query))
    # Transform the query using the TF-IDF vectorizer
    query_tfidf = vectorizer.transform([query_processed])
    # Compute the cosine similarity between the query and each sentence in the corpus
    cosine_similarities = np.dot(query_tfidf, tfidf_matrix.T).toarray()[0]
    # Get the indices of the top_n most similar sentences
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    # Retrieve and return the original sentences
    return [original_sentences[i] for i in most_similar_indices]

def chatbot(question):
    # Find the most relevant sentences
    most_relevant_sentences = get_most_relevant_sentences(question)
    # Join the sentences into a single response
    detailed_response = ' '.join(most_relevant_sentences)
    return detailed_response

# Create a Streamlit app
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()