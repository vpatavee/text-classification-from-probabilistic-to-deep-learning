# Text Classification: From Probabilistic Models To Deep Learning

This blog is a detailed review and experiments of text classification techniques, from classical technique in 1960s
like Naive Bayes to 2010s technique like LSTM and BERT. By following these experiments, you will
get some senses of what techniques and variations you can do to solve text classification, 
intuitions behind those techniques and outcomes you should exepect. In each 
experiments, there are links to related papers, Stackoverflow discussion or textbook so that you have
comprehensive knowledges on the technical topic being discussed.

There is no hard rules for NLP. Unlike mathemetic which you can formally proof, NLP is a lot more empirical. 
There are several things you can experiment which come from your imagination and assumption.
From the experiments, you may come up with logical explanation that sounds. However, you will never be able 
to formally prove it. While working on NLP problems, you have to keep that in mind.

So why text classification is so important? I intentionally used 
word "Text Classification" not "Sentiment Analysis" for reasons, despite the sentiment
analysis dataset we will be using. The text classification is broader problem set.
We can think of sentiment analysis as a subset of text classification, 
where classes are good and bad sentiment. Generally, the real word problems are not neccesary
well-scoped, as they are created from human perspective not acadamic. 
What we usually do is to reduce those problems into
some problem sets that are well-scoped that we know how to solve. Fortunately, lots of 
NLP problems in real world can be reduced to text classification. The very basic examples
that are shown up in almost every NLP / ML textbook include
- sentiment analysis e.g. movie reviews, food reviews, product reviews
- spam detection

However, there are a lot more. For example
- query classification - imagine you have two different seach systems i.e. different indices, different 
information retrieval techniques that designed for different kind of searches. One simple exmaple is
Google Search. It has to determine if a query is a question which it can get the answer from Google 
Knowledge Graph and answer the question directly, or it's keywords search which will retrieve
the documents that are likely to contain the answer. In order to route
queries to right system, we need to discover the intent of the users. The intent discovery can be reduced to
text classification technique.
- text scoring - this problems may include student essay grading (grade), urgency assignments (number of days),
and so on. Since the prediction target is continuous, we can use regression techniques. However, 
most of the time the classification technique perform better when the features are text.
We can reduce this problem set to text classification by cutting the continous target into ranges.
This simplification is usually good enough for most industrial problems.


**Dataset**

We will use [IMDB Review](https://www.tensorflow.org/datasets/catalog/imdb_reviews) in this experiment.

**Do and Don't**

Do
- experiment and discuss on text classifcation techniques in details which you can apply to your own problems
- review related NLP theory and provide links for further study
- provide extensive use of most common ML/NLP libraries i.e. spaCy, sklearn, tensorflow, gensim etc.

Don't
- introduce novel text classification techniques

**Target Audience**
- If you have background in ML but not NLP, you may follow each experiment one by one
and follow the link to optional readings. At the end, you will learn fundamental concept
of text classifcation problems are related NLP area.
- If you have background in ML and NLP, you may jump into specfic notebooks or experiements
that you are interested in particular.
    
**Environmental Setup**

First, clone this reposiory to your working environment. Make sure you have have Python 3.6 installed.
Although these Notebooks were developed in Python 3.6, earler Python 3 might also work. 

1. create virtual environment (not required, but highly recommended). In you working directory,
do `python3 -m venv venv` and activate virtual environment `source venv/bin/activate`. 
2. install prerequisite `pip install -r requiresment.txt`
3. download spaCy model `python -m spacy download en_core_web_sm`
4. some Notebooks e.g. Notebook 2,3 and 4, may require special library. Follow the instruction in the Notebooks.
    
This block is organized into 5 chapters.
    
**0. GTKY**
Most data scientist miss this very important GTKY step. They just download the data and rush into building models. 
However, when you do empirical study, it is very important to know the data. 
You may come up with intuitions from just a glance. In this Notebook, 
we will get to know our dataset as well as discuss some text preprocessing techniques we can use.


**1. Classical Machine Learning Models**

In this Notebook, we will focus on tokenization, vectorization (how to featurize text)
and classical classification models i.e. Naive Bayes and Logistic Regression.


**2. Word Embeddings**

In this Notebook, we will jump to early 2000s era when the word embeddings is introduced.
We will also experiment vectorization techniques which is different from Notebook 1.
We will use pre-trained word embeddings model: GloVE and Word2Vec as input of our text classifcation.

**3. More on Word Embeddings**

In this Notebook, we will not use pre-trained word embeddings, but will train it from scrath (you
will se why it's necessary). We will focus on model hyperparameters in their effect. We will also
try transfer learning concept on our word embeddings training process.


**4. BERT**

TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO

**5. GPT-2**

TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
TODO TODO TODO TODO TODO TODO TODO TODO
