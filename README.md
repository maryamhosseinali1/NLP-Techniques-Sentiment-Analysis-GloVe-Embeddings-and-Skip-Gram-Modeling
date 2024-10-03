# **NLP Techniques: Sentiment Analysis, GloVe Embeddings, and Skip-Gram Modeling**

## **Project Overview**
This NLP project is divided into three main parts:
1. Sentiment Analysis using Naïve Bayes with different vectorization techniques.
2. Sarcasm Detection utilizing GloVe word embeddings and logistic regression.
3. Word Vector Generation using the Skip-Gram model from word2vec, exploring word relationships.

The project explores how different embedding techniques and models affect text classification and semantic understanding.

## **Table of Contents**
1. [Objectives](#objectives)
2. [Sentiment Analysis with Naïve Bayes](#sentiment-analysis-with-naïve-bayes)
3. [Sarcasm Detection with GloVe Embeddings](#sarcasm-detection-with-glove-embeddings)
4. [Skip-Gram Model for Word Vectors](#skip-gram-model-for-word-vectors)
5. [Results and Evaluation](#results-and-evaluation)
6. [Tools and Dependencies](#tools-and-dependencies)


## **Objectives**
The primary objectives of this project are:
- To classify sentiments in tweets using Naïve Bayes and vectorization techniques like TF, TF-IDF, and PPMI.
- To detect sarcasm in headlines using GloVe embeddings for semantic representation.
- To build a Skip-Gram word2vec model to generate word vectors and explore semantic word relationships.

---

## **Sentiment Analysis with Naïve Bayes**
### **Task**:
Classify the sentiment of tweets as positive or negative using Naïve Bayes with vectorization methods like Term Frequency (TF), Term Frequency-Inverse Document Frequency (TF-IDF), and Positive Pointwise Mutual Information (PPMI).

### **Key Steps**:
1. **Dataset Preprocessing**: Loaded the Sentiment140 dataset and cleaned tweets (e.g., lowercasing, removing URLs and punctuation, tokenization, stopword removal, stemming).
2. **Feature Extraction**:
   - **TF Vectorization**: Created binary term frequency vectors for the presence of words.
   - **TF-IDF Vectorization**: Generated vectors considering the term frequency and inverse document frequency.
   - **PPMI Vectorization**: Applied PPMI to capture associations between words.
3. **Model Training and Evaluation**: Trained a Naïve Bayes classifier on each vectorization method and evaluated based on F1-score, Precision, and Recall.

### **Results Summary**:
- **TF**: Best balance in F1-score.
- **TF-IDF**: Slightly lower performance, indicating term rarity may not enhance sentiment detection.
- **PPMI**: Improved recall but overall performance similar to TF.

---

## **Sarcasm Detection with GloVe Embeddings**
### **Task**:
Identify sarcasm in news headlines using GloVe word embeddings and logistic regression.

### **Key Steps**:
1. **Dataset Preprocessing**: Cleaned headlines by removing punctuation, special characters, and stopwords.
2. **GloVe Embedding Preparation**: Loaded pre-trained GloVe vectors (50d, 100d, 200d, 300d) and tokenized headlines for embedding.
3. **Feature Transformation**: Converted each headline into a GloVe-averaged vector.
4. **Model Training and Evaluation**: Trained logistic regression models on different embedding dimensions and evaluated them based on F1-score, Precision, and Recall.

### **Results Summary**:
- **Performance Improvement with Higher Dimensions**: Increasing GloVe dimensions from 50d to 300d improved model accuracy and semantic understanding.

---

## **Skip-Gram Model for Word Vectors**
### **Task**:
Generate word embeddings using the Skip-Gram model and explore word similarities.

### **Key Steps**:
1. **Data Preparation**: Cleaned and tokenized text data (e.g., Sherlock Holmes stories).
2. **Skip-Gram Model with Negative Sampling**:
   - Generated skip-grams (word pairs in context) and negative samples for training.
   - Built and trained a Skip-Gram model using a target-context approach.
3. **Word Vector Analysis**:
   - **Semantic Arithmetic**: Demonstrated word relationships using vector arithmetic (e.g., king - man + woman ≈ queen).
   - **Visualization**: Used PCA to reduce embedding dimensions and visualized relationships between words like "brother-sister," "uncle-aunt."

### **Results Summary**:
- **Semantic Relationships Captured**: The model effectively captured relationships between words, with high cosine similarity for semantically related words.

---

## **Results and Evaluation**
- **Sentiment Analysis**: Naïve Bayes showed reasonable performance across vectorization methods, with TF slightly outperforming others.
- **Sarcasm Detection**: Logistic regression with GloVe embeddings achieved higher accuracy as embedding dimensions increased.
- **Skip-Gram Word Embeddings**: The model captured meaningful semantic relationships, as demonstrated by vector arithmetic and PCA visualizations.

---

## **Tools and Dependencies**
- **Python Libraries**: `pandas`, `nltk` (text processing), `scikit-learn` (modeling), `tensorflow.keras` (GloVe embeddings), `gensim` (word2vec).
- **Data Sources**: Sentiment140 for sentiment analysis, Sarcasm detection dataset for headline classification, textual data for Skip-Gram model training.

---



