# Text Classification: From Probabilistic Models To Deep Learning

This blog is a detailed review and experiments of text classification techniques, from classical technique in 1960s
e.g. Naive Bayes to 2010s technique e.g. LSTM and BERT. By following these experiments, you will
get some sense of what techniques and variations you can do to solve text classification, 
the intuitions behind those techniques and outcomes you should expect. 
This blog also provides the code samples that you can apply to your own text classification problems.
Inside the Notebooks, there are set of experiments, theory behind them and external resources
e.b. links to related papers, Stack Overflow or textbook so that you have
comprehensive knowledges on the technical topic being discussed.

There is no hard rule for NLP. Unlike mathematical problems which you can formally proof, NLP is a lot more empirical. 
It opens for your creativities to explore new notions, setup the experiments and test them.
Then, you may come up with logical explanation that supports the results of your experiments. 
However, you will never be able to formally prove it. While working on NLP problems, you have to keep that in mind.

So why are text classification problems so important? I intentionally named this blog as
"Text Classification" not "Sentiment Analysis", despite the sentiment
analysis dataset used in this blog. The text classification is broader problem set.
You can think of sentiment analysis as a subset of text classification, 
where classes are sentiments. In fact, the real world problems are not necessary 
well-scoped, as they are created from human perspective not from academic. 

What we usually do is to reduce those problems into
some problem sets that are well-scoped, and we know how to solve. Fortunately, lots of 
NLP problems in real world can be reduced to text classification. The very basic examples
that are shown up everywhere include
- sentiment analysis e.g. movie reviews, food reviews, product reviews
- spam detection

However, there are many more. For example
- query classification - imagine you have two different search systems which support different query intentions. For example, Google Search has to determine if a query is a question or key words search. If it is a question, it will answer that question directly. Otherwise. it will retrieve the documents that are likely to contain the answer. In order to route queries to right system, we need to discover the intent of the users. The intent discovery can be reduced to text classification problem.
- text scoring - for example automated essay gradings, urgency level assignments, etc. Since the prediction target is continuous, we can use regression techniques. However, the classification techniques usually perform better when the features are text. We can reduce this problem set to multiclasses text classification by cutting the continuous target into small ranges. This simplification is good enough for most problems.

This blog focuses on:

- empirical studies and detailed discussions on broad range of text classification techniques which you can apply to your own problems
- guideline to approach NLP problems systematically
- reviews of related NLP topics and links to external resources 
- code samples for most common ML/NLP libraries i.e. spaCy, sklearn, tensorflow, gensim 


This blog does NOT focus on:
- "novel" text classification techniques
- beating the benchmark for this particular dataset. For IMDB dataset leaderboards, check out [this](http://nlpprogress.com/english/sentiment_analysis.html) and [this](https://paperswithcode.com/sota/sentiment-analysis-on-imdb)


## Dataset
We will use [IMDB Review](http://ai.stanford.edu/~amaas/data/sentiment/) in this experiment. It consists of 25,000 highly polar movie reviews for training and 25,000 for testing. 



## Target Audience
- If you have background in ML but not NLP, you may follow each experiment step by step
and follow the link to external resources throughout the Notebooks. At the end, you will learn fundamental concept
of text classifcation problems and other related NLP topics.
- If you have background in ML and NLP, you may jump into specfic notebooks or experiments
that you are interested in particular.
    
## Environmental Setup
First, clone this repository to your working environment. Make sure you have Python 3.6 installed.
Although these Notebooks were developed in Python 3.6, earlier Python 3 might also work. 

1. create virtual environment (not required, but highly recommended). In you working directory,
do `python3 -m venv venv` and activate virtual environment `source venv/bin/activate`. 
2. install prerequisite `pip install -r requiresment.txt`
3. download spaCy model `python -m spacy download en_core_web_sm`
4. some Notebooks may require special library or pre-trained models. Follow the special instruction inside the Notebooks.


## Notebooks
    
**0. GTKY**

Most data scientists miss this very first but important GTKY (Get To Know You) step. 
They usually just download the data and jump into models development. 
However, when you do experiments, it is crucial to know the dataset very well. 
You may come up with intuitions from by just giving dataset a glance. In this Notebook, 
I will walk you through the IMDB dataset, as well as discuss some essential text preprocessing techniques. 
In addition, before start building machine learning models in the following Notebooks, let's create a simple 
rule-based model. Let's say giving positive review if word "good", "fantastic" or "awesome" is in the review, otherwise giving negative. 
Set this as the baseline model. Most industrial NLP problems can be solved with decent accuracy using simple rule based model.


**1. Classical Machine Learning**

We will discuss classical classification models i.e. Naive Bayes and Logistic Regression. Since it's the first Notebook, I will walk you through the various tokenization and vectorization processes. The following Notebooks will skip this parts.

**2. Word Embeddings**

We will jump from 1960s to 2000s when the Word Embeddings was introduced. We will start with the brief summary of Word Embeddings and provide links to external resources. Then, we will discuss several approaches we can make use of word embeddings for text classification and the theory behind them. In this Notebook, we will use most popular pre-trained Word Embeddings namely Word2Vec an GloVE.


**3. More on Word Embeddings**

We will go beyond pre-trained Word Embeddings we use in Notebook 2 by training it from scratch (we will discuss why we should do that inside the Notebook). We will experiment several Word Embeddings model hyperparameters and their effect and attach links to related research papers. By the end of the Notebook, we will experiment [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning), a technique that stores knowledge gained while solving one problem and applying it to a different but related problem, on the word embeddings.


**4. Deep Learning**

Deep Learning has been widely used for solving NLP problems since early 2010s. In this Notebook, we will use variety of LSTM and CNN based model on our dataset and discuss some theory behind the models as usual.

**5. More on Deep Learning**

In 2017, after the publication of research paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the trend has been shifting from LSTM based model to transformer based language models, for example, [BERT](https://arxiv.org/abs/1810.04805) and [GPPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Check out [this](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) on how we can use BERT language model to solve IMDB dataset.


