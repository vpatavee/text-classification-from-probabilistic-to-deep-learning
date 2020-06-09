import spacy
import pickle
import numpy as np
from collections import Counter
from spacy.tokens import DocBin
import os
import hashlib
import time
import re

nlp = spacy.load("en_core_web_sm")
SPACY_DOC_PATH = "tmp/spacy_doc"

TAG_RE = re.compile(r'<[^>]+>')
SYM = re.compile(r'[_,.\-()?]+')
EXCLAIM = re.compile(r'!')


def remove_html_tags(text):
    return TAG_RE.sub("", text)


def remove_some_symbols(text):
    text =  re.sub(SYM, " ", text)
    text =  re.sub(EXCLAIM, " ! ", text)
    return text

def preprocess(sents):
    return [remove_some_symbols(remove_html_tags(text)) for text in sents]


def load_or_create_spacy_doc(sents, do_preprocess, use_cache, verbose):
    """
    @sents list of string to be tokenized.
    @use_cache if true, try load from disk first. Otherwise, tokenize.
    @return DocBin object
    """
    
    if do_preprocess:
        sents = preprocess(sents)
        
    fname = SPACY_DOC_PATH + str(do_preprocess) + hash_sents(sents) + ".bin"
    
    if os.path.exists(fname) and use_cache:
        now = time.time()
        if verbose:
            print("Loading tokenized document from disk...")
        with open(fname, "rb") as f:
            doc_bin = DocBin().from_bytes(f.read())
        if verbose:
            print("Finished loading tokenized document in {:.2f}s!".format(time.time() - now))
        return doc_bin
    else:
        now = time.time()
        if verbose:
            print("Start tokenizing document...")
        doc_bin = DocBin()
        for doc in nlp.pipe(sents, disable=["parser", "tagger"]):
            doc_bin.add(doc)
        with open(fname, "wb") as f:
            f.write(doc_bin.to_bytes())
        if verbose:
            print("Finish tokenizing document and save to disk in {:.2f}s!".format(time.time() - now))
        return doc_bin

def is_ignore(tok, ignore):
    for ignore_attr in ignore:
        if getattr(tok, ignore_attr):
            return True
    return False

def hash_sents(sents):
    m = hashlib.md5()
    for e in sents:
        m.update(e.encode('utf-8')) 
    return  m.hexdigest()

def spacy_tokenizer(sents, lower=False, lemma=False, ignore=None, use_cache=True, do_preprocess=True, verbose=False, **kwargs):          
    # use_cache 70 s, not use_cache 327 s
    doc_bin = load_or_create_spacy_doc(sents, do_preprocess, use_cache, verbose)
    docs = list()
    for doc_obj in doc_bin.get_docs(nlp.vocab):
        doc = list()
        for tok_obj in doc_obj:
            if ignore and is_ignore(tok_obj, ignore):
                continue
            if lemma:
                tok = tok_obj.lemma_
            else:
                tok = tok_obj.orth_
            if lower:
                doc.append(tok.lower())
            else:
                doc.append(tok)
        docs.append(doc)
    return docs


def print_stat(list_of_text, models=None):
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

    vocabs = set(counter.keys())
    if models:
        for model_name, model in models.items():
            num_oov = len(vocabs) - len(vocabs & set(model.vocab.keys()))
            print("word embedding model: {}, num oov: {}, percent of oov: {:.2f}".format(model_name, num_oov, num_oov/ len(vocabs))) 

    print("\nexample")
    len_ = len(list_of_text)
    print(list_of_text[len_//2])
    print()
    print(list_of_text[len_//4])
    
    
def print_words_freq_stat(tokenized_sents, n=50):
    counter = Counter([e for sent in tokenized_sents for e in sent])
    once_occurence = len([e for e in counter if counter[e] ==1])
    num_unique_tokens = len(counter)
    
    print("number of unique tokens", num_unique_tokens)
    print("number of tokens occuring only once", once_occurence)
    print("fraction of tokens occuring only once: %0.2f" % (once_occurence / num_unique_tokens) )
    print("least {} common words and their occurrenceL:\n{}".format(str(n), str(counter.most_common()[-n:])))

    
