from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nlp_utils import preprocess_remove_html_non_ascii, preprocess_remove_html, spacy_tokenizer, spacy_tokenizer_lower, \
    spacy_tokenizer_lower_lemma, spacy_tokenizer_lower_lemma_remove_stop, spacy_tokenizer_lower_lemma_remove_stop_and_punc
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd


def create_count_vects(preprocessors, tokenizers, count_vect_classes, is_binary, ngrams):
    """
    create a dict of CountVectorizer object
    key = tuple (sklearn_vectorizer_class, preprocesser, tokenizer, ngrams, binary)
    value = CountVectorizer object coresponding to the config in key
    """
    count_vects = dict()
        
    for pre in preprocessors:
        for tok in tokenizers:
            for ngram in ngrams:
                for binary in is_binary:
                    for count_vect_class in count_vect_classes:
                        key = (
                            count_vect_class.__name__,
                            pre.__name__,
                            tok.__name__,
                            str(ngram),
                            str(binary)
                        )
                        count_vects[key] = count_vect_class(
                            analyzer="word", 
                            ngram_range=ngram,
                            tokenizer=tok, 
                            preprocessor=pre,
                            binary=binary
                        ) 

    return count_vects


def run_logistic_exp(X_train, X_test, y_train, y_test):

    preprocessors = [preprocess_remove_html_non_ascii]
    tokenizers = [
        spacy_tokenizer, 
        spacy_tokenizer_lower_lemma
    ]
    count_vect_classes = [CountVectorizer, TfidfVectorizer]
    is_binary = [True, False]    
    ngrams = [(1,1), (1,2)]

    count_vects = create_count_vects(preprocessors, tokenizers, count_vect_classes, is_binary, ngrams)
    res = dict()

    for name, count_vect in count_vects.items():
        X_train_counts = count_vect.fit_transform(X_train)
        X_test_counts = count_vect.transform(X_test)
        best_f1 = 0
        for c in range(-3,3):
            C = 10 ** c
            clf = LogisticRegression(random_state=0, max_iter=1000, C=C)
            clf.fit(X_train_counts, y_train)

            y_test_pred = clf.predict(X_test_counts)
            f1 = f1_score(y_test, y_test_pred)
            best_f1 = max(f1, best_f1)
            
        res[name] = best_f1

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index()
    res_df.columns = ["index", "F1"]
    res_df["vectorizer"] = res_df["index"].apply(lambda x: x[0])
    # res_df["preprocessing"] = res_df["index"].apply(lambda x: x[1])
    res_df["tokenizer"] = res_df["index"].apply(lambda x: x[2])
    res_df["ngram"] = res_df["index"].apply(lambda x: x[3])
    res_df["binary/multinomial"] = res_df["index"].apply(lambda x: x[4])
    return res_df[["vectorizer", "tokenizer", "ngram", "binary/multinomial", "F1"]]
                

def run_multi_nb_exp(X_train, X_test, y_train, y_test):

    preprocessors = [preprocess_remove_html_non_ascii]
    tokenizers = [
        spacy_tokenizer, 
        spacy_tokenizer_lower_lemma,
        spacy_tokenizer_lower_lemma_remove_stop,
    ]
    count_vect_classes = [CountVectorizer, TfidfVectorizer]
    is_binary = [False]  # Multinomial NB takes numeric features
    ngrams = [(1,1), (1,2)]

    count_vects = create_count_vects(preprocessors, tokenizers, count_vect_classes, is_binary, ngrams)
    res = dict()

    for name, count_vect in count_vects.items():
        X_train_counts = count_vect.fit_transform(X_train)
        X_test_counts = count_vect.transform(X_test)

        clf = MultinomialNB()
        clf.fit(X_train_counts, y_train)

        y_test_pred = clf.predict(X_test_counts)
        f1 = f1_score(y_test, y_test_pred)            
        res[name] = f1

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index()
    res_df.columns = ["index", "F1"]
    res_df["vectorizer"] = res_df["index"].apply(lambda x: x[0])
    # res_df["preprocessing"] = res_df["index"].apply(lambda x: x[1])
    res_df["tokenizer"] = res_df["index"].apply(lambda x: x[2])
    res_df["ngram"] = res_df["index"].apply(lambda x: x[3])
    res_df["binary/multinomial"] = res_df["index"].apply(lambda x: x[4])
    return res_df[["vectorizer", "tokenizer", "ngram", "binary/multinomial", "F1"]]


def run_ber_nb_exp(X_train, X_test, y_train, y_test):

    preprocessors = [preprocess_remove_html_non_ascii]
    tokenizers = [
        spacy_tokenizer, 
        spacy_tokenizer_lower_lemma,
        spacy_tokenizer_lower_lemma_remove_stop,
    ]

    count_vect_classes = [CountVectorizer] # Bernoulli NB takes only binary feature
    is_binary = [True]  # Bernoulli NB takes only binary feature
    ngrams = [(1,1), (1,2)]

    count_vects = create_count_vects(preprocessors, tokenizers, count_vect_classes, is_binary, ngrams)
    res = dict()

    for name, count_vect in count_vects.items():
        X_train_counts = count_vect.fit_transform(X_train)
        X_test_counts = count_vect.transform(X_test)

        clf = BernoulliNB()
        clf.fit(X_train_counts, y_train)

        y_test_pred = clf.predict(X_test_counts)
        f1 = f1_score(y_test, y_test_pred)            
        res[name] = f1

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index()
    res_df.columns = ["index", "F1"]
    res_df["vectorizer"] = res_df["index"].apply(lambda x: x[0])
    # res_df["preprocessing"] = res_df["index"].apply(lambda x: x[1])
    res_df["tokenizer"] = res_df["index"].apply(lambda x: x[2])
    res_df["ngram"] = res_df["index"].apply(lambda x: x[3])
    res_df["binary/multinomial"] = res_df["index"].apply(lambda x: x[4])
    return res_df[["vectorizer", "tokenizer", "ngram", "binary/multinomial", "F1"]]