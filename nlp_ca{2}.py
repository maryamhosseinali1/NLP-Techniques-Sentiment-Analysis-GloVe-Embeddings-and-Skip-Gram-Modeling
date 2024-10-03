# -*- coding: utf-8 -*-
"""NLP-CA{2}-610398209.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fQEnSFtVrTDjSmoVpgK2ApH2zTU9slDQ
"""

from google.colab import drive
drive.mount('/content/drive')

csv_file_path = '/content/drive/My Drive/training.1600000.processed.noemoticon.csv'

import pandas as pd


csv_file_path = '/content/drive/My Drive/training.1600000.processed.noemoticon.csv'

# loading dataset
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1', names=['target', 'ids', 'date', 'flag', 'user', 'text'])

data.sample(5)

"""## **part1**"""

# choosing 5000 negative and 5000 positive tweets randomly

negative_samples = data[data['target'] == 0].sample(n=5000)
positive_samples = data[data['target'] == 4].sample(n=5000)

# concat negative and positive samples into sampled_data
sampled_data = pd.concat([negative_samples, positive_samples]).reset_index(drop=True)

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
	# converting text to lowercase
	text = text.lower()

	# removing URLs
	text = re.sub(r'http\S+|www.\S+', '', text)

	# removing punctuation
	translator = str.maketrans('', '', string.punctuation)

	text = text.translate(translator)

	# tokenization
	words = word_tokenize(text)

	# removing stopwords
	filtered_words = [word for word in words if word not in stopwords.words('english')]

	# stemming
	stemmer = PorterStemmer()
	stemmed_words = [stemmer.stem(word) for word in filtered_words]

	# rejoin words to form the processed text
	return ' '.join(stemmed_words)

sampled_data['processed_text'] = sampled_data['text'].apply(preprocess_text)

print(sampled_data[['text', 'processed_text']].head())

from sklearn.model_selection import train_test_split

# splitting the dataset
X = sampled_data['text']  # features
y = sampled_data['target']  # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## **part2**"""

import pandas as pd

# converting the processed_text column to a list of documents
documents = sampled_data['processed_text'].tolist()

# creating a sorted list of unique words in the documents to form the vocabulary
vocabulary = sorted(set(word for document in documents for word in document.split()))

# an empty list for binary TF vectors
binary_tf_matrix = []

 # binary TF matrix
for document in documents:

    # a set of unique words in the current document
    words_in_document = set(document.split())

    # setting the binary TF vector with zeros
    binary_tf_vector = [0] * len(vocabulary)

    # iterating over vocab and when a word is present in the document,
    # setting the corresponding value to 1
    for index, word in enumerate(vocabulary):
        if word in words_in_document:
            binary_tf_vector[index] = 1


    binary_tf_matrix.append(binary_tf_vector)

# converting the binary TF matrix to a DataFrame
binary_tf_df = pd.DataFrame(binary_tf_matrix, columns=vocabulary)



"""##**part** **5** **for** **Term** **Frequency**"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np


# re-defining vocabulary
vocabulary = sorted(set(word for document in sampled_data['processed_text'].tolist() for word in document.split()))

# if 'binary_tf_df' has not been created , recreate it here
binary_tf_matrix = []

for document in sampled_data['processed_text'].tolist():
    words_in_document = set(document.split())
    binary_tf_vector = [1 if word in words_in_document else 0 for word in vocabulary]
    binary_tf_matrix.append(binary_tf_vector)

binary_tf_df = pd.DataFrame(binary_tf_matrix, columns=vocabulary)

# align 'binary_tf_df' with labels 'y'
X = binary_tf_df.values
y = sampled_data['target']  # Ensure label column

# splitting the dataset into training and testing sets
X_train_tf, X_test_tf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize and train the Naive Bayes classifier
clf_tf = MultinomialNB()
clf_tf.fit(X_train_tf, y_train)

# predict on the test set and evaluate
y_pred_tf = clf_tf.predict(X_test_tf)

# as'4' is the label for the positive class
print("Performance with TF Embedding:")
print(f"F1-score: {f1_score(y_test, y_pred_tf, pos_label=4)}")
print(f"Precision: {precision_score(y_test, y_pred_tf, pos_label=4)}")
print(f"Recall: {recall_score(y_test, y_pred_tf, pos_label=4)}")

"""## **part3**"""

import math
import pandas as pd
from collections import defaultdict

documents = sampled_data['processed_text'].tolist()

# function to calculate TF
def compute_tf(document):
    tf_dict = defaultdict(int)
    words = document.split()
    for word in words:
        tf_dict[word] += 1
    total_terms = len(words)
    for word, count in tf_dict.items():
        tf_dict[word] = count / total_terms
    return tf_dict

# Function to calculate IDF
def compute_idf(documents):
    idf_dict = defaultdict(int)
    N = len(documents)
    for document in documents:
        for word in set(document.split()):  # unique words
            idf_dict[word] += 1
    for word, count in idf_dict.items():
        idf_dict[word] = math.log(N / count)
    return idf_dict

# Function to calculate TF-IDF
def compute_tfidf(documents):
    idfs = compute_idf(documents)
    tfidf_matrix = []

    for document in documents:
        tf_dict = compute_tf(document)
        tfidf_vector = {word: tf * idfs[word] for word, tf in tf_dict.items()}
        tfidf_matrix.append(tfidf_vector)

    return tfidf_matrix
# computing the TF-IDF matrix
tfidf_matrix = compute_tfidf(documents)

# convert matrix into a dense matrix
# creating a sorted list of unique words in the documents to form the vocabulary
vocabulary = sorted(set(word for document in documents for word in document.split()))

# create a mapping of vocabulary words to their indexes
vocab_index = {word: i for i, word in enumerate(vocabulary)}

# a matrix to store the TF-IDF vectors
dense_tfidf_matrix = []
for doc_vector in tfidf_matrix:

    doc_tfidf_vector = [0] * len(vocabulary)
    for word, tfidf_value in doc_vector.items():
        index = vocab_index[word]
        doc_tfidf_vector[index] = tfidf_value
    dense_tfidf_matrix.append(doc_tfidf_vector)

# Convert the dense TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=vocabulary)

"""##**part** **5** **for** **TF-IDF**"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score



# align 'tfidf_df' with labels 'y'
X = tfidf_df.values
y = sampled_data['target']  # Ensure this is your label column

# pslitting the dataset into training and testing sets
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize and train the Naive Bayes classifier with TF-IDF vectors
clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)

# Predict on the test set and evaluate
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

# as the label positive class is '4'
print("Performance with TF-IDF Embedding:")
print(f"F1-score: {f1_score(y_test, y_pred_tfidf, pos_label=4)}")
print(f"Precision: {precision_score(y_test, y_pred_tfidf, pos_label=4)}")
print(f"Recall: {recall_score(y_test, y_pred_tfidf, pos_label=4)}")

"""## **part4**"""

import math
import numpy as np
from collections import Counter, defaultdict

documents = sampled_data['processed_text'].tolist()

# a function to compute PPMI
def compute_ppmi(documents):
    word_counts = Counter() #frequency of each word
    co_occurrences = defaultdict(Counter) # how often pairs of words co-occur
    total_co_occurrences = 0 #count of all word pairs

    # calculate word frequencies and co-occurs
    for document in documents:
        words = document.split()
        for i, word in enumerate(words):
            word_counts[word] += 1
            for j in range(max(0, i - 5), min(i + 5 + 1, len(words))):
                if i != j:
                    co_occurrences[word][words[j]] += 1
                    total_co_occurrences += 1

    # compute PPMI
    ppmi_matrix = defaultdict(dict)
    for word, contexts in co_occurrences.items():
        for context_word, co_occurrence in contexts.items():
            pmi = math.log2((co_occurrence / total_co_occurrences) / ((word_counts[word] / len(word_counts)) * (word_counts[context_word] / len(word_counts))))
            ppmi = max(pmi, 0)
            ppmi_matrix[word][context_word] = ppmi

    return ppmi_matrix, word_counts, co_occurrences

ppmi_matrix, word_counts, co_occurrences = compute_ppmi(documents)

# a sorted list of unique words
vocabulary = sorted(word_counts.keys())
vocab_index = {word: i for i, word in enumerate(vocabulary)}


dense_ppmi_matrix = np.zeros((len(documents), len(vocabulary)))
# fill matrix with PPMI values
for i, document in enumerate(documents):
    words = document.split()
    for word in words:
        if word in vocab_index:  # only consider words in the vocabulary
            for context_word, ppmi_value in ppmi_matrix[word].items():
                if context_word in vocab_index:
                    j = vocab_index[context_word]
                    dense_ppmi_matrix[i, j] = max(dense_ppmi_matrix[i, j], ppmi_value)  # use the max PPMI value

# converting the dense PPMI matrix to a DataFrame
ppmi_df = pd.DataFrame(dense_ppmi_matrix, columns=vocabulary)

"""##**part** **5** **for** **PPMI**"""

X_ppmi = ppmi_df.values
y = sampled_data['target'].values  # ensure label column

# splitting the dataset into training and testing sets
X_train_ppmi, X_test_ppmi, y_train, y_test = train_test_split(X_ppmi, y, test_size=0.2, random_state=42)

# initialize and train the Naive Bayes classifier with PPMI vectors
clf_ppmi = MultinomialNB()
clf_ppmi.fit(X_train_ppmi, y_train)

# predict on the test set and evaluate
y_pred_ppmi = clf_ppmi.predict(X_test_ppmi)
# # as the label positive class is '4'
print("Performance with PPMI Embedding:")
print(f"F1-score: {f1_score(y_test, y_pred_ppmi, pos_label=4)}")
print(f"Precision: {precision_score(y_test, y_pred_ppmi, pos_label=4)}")
print(f"Recall: {recall_score(y_test, y_pred_ppmi, pos_label=4)}")