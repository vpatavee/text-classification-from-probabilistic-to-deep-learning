from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])

def remove_html(text):
    soup = BeautifulSoup(text)
    return soup.get_text(separator=" ")

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def preprocess_remove_html_non_ascii(text):
    text = remove_html(text)
    text = remove_non_ascii(text)
    return text.strip()

def preprocess_remove_html(text):
    text = remove_html(text)
    return text.strip()

def spacy_tokenizer(sent):
    return [e.orth_ for e in nlp(sent)]

def spacy_tokenizer_lower(sent):
    return [e.orth_.lower() for e in nlp(sent)]

def spacy_tokenizer_lower_lemma(sent):
    return [e.lemma_.lower() for e in nlp(sent)]

def spacy_tokenizer_lower_lemma_remove_stop(sent):
    return [e.lemma_.lower() for e in nlp(sent) if not e.is_stop]

def spacy_tokenizer_lower_lemma_remove_stop_and_punc(sent):
    return [e.lemma_.lower() for e in nlp(sent) if not e.is_stop and not e.is_punct]

def create_count_vects():
    """
    create a dict of CountVectorizer object
    key = <preprocesser>_<tokenizer>_<ngrams>_<binary>
    value = CountVectorizer object coresponding to the config in key
    """
    count_vects = dict()
    
    preprocessors = [preprocess_remove_html_non_ascii, preprocess_remove_html]
    
    tokenizers = [
        spacy_tokenizer, 
        spacy_tokenizer_lower, 
        spacy_tokenizer_lower_lemma,
        spacy_tokenizer_lower_lemma_remove_stop,
        spacy_tokenizer_lower_lemma_remove_stop_and_punc
    ]
    
    ngrams = [(1,1), (1,2)]
    
    for pre in preprocessors:
        for tok in tokenizers:
            for ngram in ngrams:
                count_vect_name = "Count_{}_{}_{}_{}".format(
                    pre.__name__,
                    tok.__name__,
                    str(ngram),
                    "True"
                )
                
                count_vects[count_vect_name] = CountVectorizer(
                    analyzer="word", 
                    ngram_range=ngram,
                    tokenizer=tok, 
                    preprocessor=pre,
                    binary=True
                ) 

                count_vect_name = "Count_{}_{}_{}_{}".format(
                    pre.__name__,
                    tok.__name__,
                    str(ngram),
                    "False"
                )
                
                count_vects[count_vect_name] = CountVectorizer(
                    analyzer="word", 
                    ngram_range=ngram,
                    tokenizer=tok, 
                    preprocessor=pre,
                    binary=False
                ) 
                count_vect_name = "Tfidf_{}_{}_{}_{}".format(
                    pre.__name__,
                    tok.__name__,
                    str(ngram),
                    "True"
                )
                
                count_vects[count_vect_name] = TfidfVectorizer(
                    analyzer="word", 
                    ngram_range=ngram,
                    tokenizer=tok, 
                    preprocessor=pre,
                    binary=True
                ) 

                count_vect_name = "Tfidf_{}_{}_{}_{}".format(
                    pre.__name__,
                    tok.__name__,
                    str(ngram),
                    "False"
                )
                
                count_vects[count_vect_name] = TfidfVectorizer(
                    analyzer="word", 
                    ngram_range=ngram,
                    tokenizer=tok, 
                    preprocessor=pre,
                    binary=False
                )                 
    return count_vects


def convert_to_my_cs544_dataset_format_and_save(X_train, X_test, y_train, y_test, folder="IMDB_cs544_format"):
    import os
    labels = ["Neg", "Pos"]
    labels_dummy = ["True", "Fake"]
    
    train_formatted = "\n".join(
        "{} {} {} {}".format(str(i), labels_dummy[i%2], labels[yi], Xi) 
        for i, (Xi, yi) in enumerate(zip(X_train, y_train))
    )
        
    test_formatted = "\n".join(
        "{} {}".format(str(i), Xi) 
        for i, Xi in enumerate(X_test)
    )
        
    test_key = "\n".join(
        "{} {} {}".format(str(i),labels_dummy[i%2] , labels[yi]) 
        for i, yi in enumerate(y_test)
    )
        
    os.mkdir(folder)
    with open(folder + "/train.txt", "w") as f:
        f.write(train_formatted)
    with open(folder + "/test.txt", "w") as f:
        f.write(test_formatted)
    with open(folder + "/test_key.txt", "w") as f:
        f.write(test_key)        
        
    print("save into cs544_dataset_format")

    