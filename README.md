# Sentiment Analysis: From Probabilistic Models To Deep Learning

Intro
- Sentiment  Analysis is a simple task
- What models to use: Probabilistic VS Deep Learning?
- Small training data
- number of parameters

## Data Set
We use [IMDB Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews).

- show examples

## Probabilistic Models

**Preprocessing**
1. Remove HTML
2. Remove HTML and Non-Ascii

**Tokenizer**
1. Spacy Tokenizer
2. Spacy Tokenizer + LowerCase
3. Spacy Tokenizer + LowerCase + Lematization
4. Spacy Tokenizer + LowerCase + Lematization + Remove Stop Words
5. Spacy Tokenizer + LowerCase + Lematization + Remove Stop Words + Remove Punctuation

**Representation**
There are several choices of text representations, as described in [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/).
1. Vectorization: Binary vs Multinomial 
2. Ngram: 1, 2

With 2 types of preprocessing, 5 types of tokenizations, 2 types of vectorization and 2 types of ngrams, we can create 80 different type of vectorizations.

**Models**
1. Logistic Regression

2. Naive Bayes





## Deep Learning Model


### Pre-trained Word Embeddings


### Train Word Embeddings


### LSTM


### Transforer

