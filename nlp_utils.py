import spacy
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter


nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
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

def spacy_tokenizer_remove_stop(sent):
    return [e.orth_ for e in nlp(sent) if not e.is_stop]

def print_stat(list_of_text):
    """
    print
    - average number of char
    - average number of tokens
    - number of vocab
    - example of sentences
    """

    print("average number of char {:.2f}".format(np.average([len(s) for s in list_of_text])))
    tokenized = [nlp(s) for s in list_of_text]
    print("average number of tokens {:.2f}".format(np.average([len(s) for s in tokenized])))

    counter = Counter([e.orth_ for s in tokenized for e in s if not e.is_stop and not e.is_punct])
    print("total number of vocab without stop words", len(counter))
    print("most common:", counter.most_common()[:5])
    print("\nexample")
    len_ = len(list_of_text)
    print(list_of_text[len_//2])
    print()
    print(list_of_text[len_//4])
    