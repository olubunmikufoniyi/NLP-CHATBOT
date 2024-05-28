The Coolar Project

# NLP Chatbot

This project is a simple Natural Language Processing (NLP) chatbot built using Python. The chatbot uses the NLTK library for text preprocessing and Scikit-learn for TF-IDF vectorization. It allows users to ask questions about the content in a text file and returns the most relevant sentences from the file.

## Features

- Tokenizes text into sentences and words.
- Removes stopwords and punctuation.
- Lemmatizes words.
- Vectorizes the text using TF-IDF.
- Computes cosine similarity to find the most relevant sentences to a user's query.
- Provides an interactive Streamlit interface for user interaction.

## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using pip:

```bash
pip install nltk streamlit scikit-learn numpy
```

Additionally, you'll need to download some NLTK resources. These can be downloaded by running the script, as it includes commands to download the required resources.

## Usage

1. **Prepare the Text File**: Place your text file (`chatbot.txt`) in the specified directory (`C:\\Users\\This PC\\Documents\\NLP Chatbot\\`).

2. **Run the Streamlit App**: Execute the script to start the Streamlit app. You can do this by running:

```bash
streamlit run your_script.py
```

Replace `your_script.py` with the name of your Python script file.

3. **Interact with the Chatbot**: Open the Streamlit interface in your browser, type your question in the input box, and press "Submit" to get a response from the chatbot.

## Code Overview

### Import Libraries

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
```

### Download NLTK Resources

```python
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Load and Preprocess the Text File

```python
file_path = 'C:\\Users\\This PC\\Documents\\NLP Chatbot\\chatbot.txt'
encodings = ['utf-8', 'latin-1', 'utf-16']

data = None
for encoding in encodings:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = f.read().replace('\n', ' ')
        break
    except UnicodeDecodeError:
        continue

if data is None:
    st.error("Failed to read the file with the available encodings.")
    st.stop()
```

### Tokenize and Preprocess the Text

```python
sentences = sent_tokenize(data)
original_sentences = sentences.copy()

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

corpus = [' '.join(preprocess(sentence)) for sentence in sentences]
```

### Initialize TF-IDF Vectorizer

```python
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
```

### Find Most Relevant Sentences

```python
def get_most_relevant_sentences(query, top_n=3):
    query_processed = ' '.join(preprocess(query))
    query_tfidf = vectorizer.transform([query_processed])
    cosine_similarities = np.dot(query_tfidf, tfidf_matrix.T).toarray()[0]
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [original_sentences[i] for i in most_similar_indices]
```

### Chatbot Function

```python
def chatbot(question):
    most_relevant_sentences = get_most_relevant_sentences(question)
    detailed_response = ' '.join(most_relevant_sentences)
    return detailed_response
```

### Streamlit App

```python
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    question = st.text_input("You:")
    if st.button("Submit"):
        response = chatbot(question)
        st.write("Chatbot: " + response)

if _name_ == "_main_":
    main()
```
