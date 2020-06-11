from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from lib.dense_vector_repr import DenseVectorizer
from lib.nlp_utils import spacy_tokenizer, hash_sents
from gensim.models import Word2Vec, KeyedVectors
import os
import time


random_state = 999

WV_MODEL_PATH = "tmp/wv"
WV_MODEL_TRANSFER_PATH = "tmp/wv_transfer"

def train_and_eval_model(X_train, X_test, y_train, y_test):
    tuned_parameters = {"C": [ 10**n for n in range(-4, 4)]}
    clf = GridSearchCV(
        LogisticRegression(
            random_state=random_state, 
            solver='sag', # default solver fail to converge
            max_iter=2000
        ),
        param_grid=tuned_parameters,
        scoring='f1'
    )
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set: ", clf.best_params_)
    print("Best F1 on development set: %0.2f" % clf.best_score_)
    y_test_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_test_pred)  
    print("F1 on test set: %0.2f" % f1)
    
    return clf


def run_pipeline(dataset, embeddings, verbose=False, **kwargs):
    now = time.time()
    
    x_train, x_test, y_train, y_test = dataset
    
    # tokenize
    X_train_tok = spacy_tokenizer(x_train, verbose=verbose, verbose=verbose, **kwargs)
    X_test_tok = spacy_tokenizer(x_test, verbose=verbose,  verbose-verbose **kwargs)

    # vectorize
    dense = DenseVectorizer(embeddings, **kwargs)
    x_train_vect = dense.fit_transform(X_train_tok, )
    x_test_vect = dense.transform(X_test_tok)

    # train model and evaluate
    clf = train_and_eval_model(x_train_vect, x_test_vect, y_train, y_test)
    
    print("time: %0.2f" % (time.time() - now))
    return clf, dense
    
    
def train_or_load_wv(corpus, use_cache=True, verbose=False, **kwargs):
    setting = "_".join(str(k) + ":" + str(v) for k,v in kwargs.items())
    hash_code = hash_sents(corpus)
    fname = WV_MODEL_PATH + setting + hash_code + ".model"
    
    size = kwargs.get("size", 300)
    window = kwargs.get("window", 5)
    iter = kwargs.get("iter", 5)

    if os.path.exists(fname) and use_cache:
        if verbose:
            print("Load Word2Vec from disk!")
        model = KeyedVectors.load(fname)
        return model
    else:  
        if verbose:
            print("Training Word2Vec...")
        tokenized_corpus = spacy_tokenizer(corpus, verbose, **kwargs)
        model = Word2Vec(
            tokenized_corpus,  
            sg=1,
            min_count=1,
            size=size,
            window=window,
            workers=4,
            iter=iter
        )
        model.wv.save(fname)
        if verbose:
            print("Finished training Word2Vec and saved to disk!")
        return model.wv
    

def train_or_load_wv_transfer(corpus, pretrained_word2vec_path, use_cache=True, verbose=False, lockf=0, n_transfer=100000, **kwargs):
    
    setting = "_".join(str(k) + ":" + str(v) for k,v in kwargs.items()) + "_lockf:{}_n_transfer:{}".format(str(lockf), str(n_transfer))
    hash_code = hash_sents(corpus)
    fname = WV_MODEL_TRANSFER_PATH + setting + hash_code + ".model"
    
    if os.path.exists(fname) and use_cache:
        if verbose:
            print("Load Word2Vec from disk!")
        model = KeyedVectors.load(fname)
        return model
    else:
        if verbose:
            print("Training Word2Vec...")
            
        # load pretrained word2vec
        pretrained_word2vec = KeyedVectors.load_word2vec_format(pretrained_word2vec_path, binary=True)

        # create union vocab set from corpus and pretrained_word2vec
        tokenized_corpus = spacy_tokenizer(corpus, verbose, **kwargs)
        corpus_vocab_set = set([e for sent in tokenized_corpus for e in sent])
        word2vec_vocab = list(pretrained_word2vec.vocab)
            
        word2vec_vocab_set = set(word2vec_vocab[:n_transfer]) 
        vocab_set = corpus_vocab_set.union(word2vec_vocab_set) # now number of augmented words is at most equal to n_transfer
        
        # keep adding until umber of augmented words = n_transfer
        i = n_transfer
        while len(vocab_set) - len(corpus_vocab_set) < n_transfer:
            vocab_set.add(word2vec_vocab[i])
            i += 1
        
        assert len(vocab_set) - len(corpus_vocab_set) == n_transfer

        size = kwargs.get("size", 300)
        window = kwargs.get("window", 5)
        iter = kwargs.get("iter", 5)

        model = Word2Vec(
            sg=1,
            min_count=1,
            size=size,
            window=window,
            workers=4
        )

        model.build_vocab([list(vocab_set)])

        assert len(model.wv.vocab) == len(vocab_set)

        model.intersect_word2vec_format(
            fname=pretrained_word2vec_path, # This seems to be very inefficient since we have to load word2vec from disk twice...
            lockf=lockf,
            binary=True
        )

        assert (model.wv["cat"] == pretrained_word2vec["cat"]).all()

        model.train(
            tokenized_corpus, 
            total_examples=len(tokenized_corpus), 
            epochs=iter
        )

        if lockf == 0:
            assert (model.wv["cat"] == pretrained_word2vec["cat"]).all()
        else:
            assert (model.wv["cat"] != pretrained_word2vec["cat"]).any()

        assert len(model.wv.vocab) == len(vocab_set)
        
        model.wv.save(fname)
        if verbose:
            print("Finished training Word2Vec and saved to disk!")
            
        return model.wv
    
    
    