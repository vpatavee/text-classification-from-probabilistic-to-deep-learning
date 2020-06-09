from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from lib.nlp_utils import spacy_tokenizer

random_state=999


def create_vectorizer_object(is_tfidf, is_bigram):
    
    # Here is little tricky. Since we plan to pass tokenized input to vectorizer,
    # every internal functions in vectorizer i.e. preprocessor, tokenizer, lowercase etc.
    # have to be override. To do so, we have to pass a callable as analyzer. This callable 
    # does nothing but simply returns itself, since we already tokenized the sentence.
    # Again, this is for the sake of runtime efficiency of our experiments. Not recommend for 
    # production level codes.
    
    if is_bigram:
        def analyzer(x):
            bigram = list()
            for i in range(len(x)-1):
                 bigram.append((x[i] + "__,__" +  x[i+1]))
            
            return x + bigram
    else:
        def analyzer(x):
            return x
        
    if is_tfidf:
        vectorizer = TfidfVectorizer(analyzer=analyzer)
    else:
        # Same thing as above
        vectorizer = CountVectorizer(analyzer=analyzer)
    
    return vectorizer

def create_model_obj(use_nb):
    if use_nb:
        model = MultinomialNB()
        tuned_parameters = dict()    
    else:
        model = LogisticRegression(
            random_state=random_state,
            max_iter=3000,
        )
        tuned_parameters = {"model__C": [ 10**n for n in range(-3, 3)]}

    return model, tuned_parameters
        
def run_pipeline(dataset, **kwargs):
    x_train, x_test, y_train, y_test = dataset
    
    # tokenize
    # Although tokenization can be done by passing spacy_tokenizer into CountVectorizer,
    # here we tokenize beforehand just for the sake of runtime efficiency of our 
    # experiments. Since we have to run several experiments, it is faster to tokenize 
    # just once and next time we just load from disk. Not recommend for production level codes.
    X_train_tok = spacy_tokenizer(x_train, **kwargs)
    X_test_tok = spacy_tokenizer(x_test,  **kwargs)
    
    is_tfidf = kwargs.get("tfidf", False)
    is_bigram = kwargs.get("bigram", False)
    use_nb =  kwargs.get("use_nb", False)
    
    # vectorize   
    vectorizer = create_vectorizer_object(is_tfidf, is_bigram)
    
    # train model and evaluate
    model, tuned_parameters = create_model_obj(use_nb)
            
    pipe = Pipeline(
        [
            ('vectorizer', vectorizer), 
            ('model', model )
        ]
    )
    
    clf = GridSearchCV(
        pipe,
        param_grid=tuned_parameters,
        scoring='f1'
    )
    
    clf.fit(X_train_tok, y_train)
        
    print("Best parameters set found on development set: ", clf.best_params_)
    print("Best F1 on development set: %0.2f" % clf.best_score_, 2)
    y_test_pred = clf.predict(X_test_tok)
    f1 = f1_score(y_test, y_test_pred)  
    print("F1 on test set: %0.2f" % f1)
    
    return clf, vectorizer
   