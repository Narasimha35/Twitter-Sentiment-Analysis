# Twitter Sentiment Analysis using NLP and Machine Learning

## Overview

This project performs sentiment analysis on Twitter data and classifies tweets into Positive, Negative, and Neutral categories using Natural Language Processing (NLP) and Machine Learning techniques.

## Dataset

* Twitter Training Dataset
* Approximately 74,000 tweets
* Classes: Positive, Negative, Neutral, Irrelevant

## Preprocessing

* Converted text to lowercase
* Removed URLs and special characters
* Removed stopwords using NLTK
* Applied Porter Stemming

## Feature Engineering

* TF-IDF Vectorization
* Maximum Features: 5000

## Machine Learning Model

* Logistic Regression

## Results

* Accuracy: 75.56%
* Precision, Recall, and F1-score evaluated on test data

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Google Colab

## Future Improvements

* Streamlit Web Application
* Word Cloud Visualization
* SVM and Naive Bayes Comparison
* Transformer-based Models (BERT)
