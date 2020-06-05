import gensim
from collections import defaultdict, Counter
import numpy as np
import random
from numpy import linalg

random.seed(999)


class PollingFunction:
    NORM = "norm"
    SUM = "sum"
    LOG = "log"

class DenseVectorizer:
    # Try to make it resemble to Scikit-learn Countvectorizer
        
    def __init__(self, model, polling=PollingFunction.NORM, tfidf=False, print_stat=True, **kwargs):
        self.model = model 
        self.model_vocab_list = list(self.model.wv.vocab)
        self.tfidf = tfidf
        self.polling = polling
        self.stat = dict()
        if self.tfidf:
            self.idf = defaultdict(int)
        else:
            self.idf = defaultdict(lambda: 1)

    def fit_transform(self, raw_documents):
        if self.tfidf:
            self._cal_idf(raw_documents)
        return self.transform(raw_documents, True)

    def print_stat(self):
        for k in self.stat:
            print(k)
            print("oov freq", self.stat[k]["num_oov"] / self.stat[k]["num_tok"])
            print("%unk in vocab", self.stat[k]["num_unique_oov"] / len(self.model_vocab_list))

    def transform(self, raw_documents, is_fit=False):
        num_tok = 0
        oov = set()
        num_oov = 0
        
        dim = self.model.vector_size
        num_doc = len(raw_documents)
        mat = np.zeros([num_doc, dim])

        for i, doc in enumerate(raw_documents):
            tf = Counter(doc)
            for w, n in tf.items():
                num_tok += n
                if w in self.model:
                    if self.polling == PollingFunction.LOG:
                        mat[i, :] += (1 + np.log(n)) * self.model[w] * self.idf[w]
                    else:
                        mat[i, :] += n * self.model[w] * self.idf[w]
                        
                else:
                    num_oov += n
                    oov.add(w)
                    for random_word in random.sample(self.model_vocab_list, k=n):
                        mat[i, :] += self.model[random_word] * self.idf[w]
                        
        if is_fit:
            self.stat["fit_transform"] = {"num_tok": num_tok, "num_oov": num_oov, "num_unique_oov": len(oov)}
        else:
            self.stat["transform"] = {"num_tok": num_tok, "num_oov": num_oov, "num_unique_oov": len(oov)}
                        
        if self.polling == PollingFunction.NORM:
            return mat / linalg.norm(mat, axis=1).reshape(-1,1)
        
        return mat

    def _cal_idf(self, raw_documents):

        num_docs = len(raw_documents)
        for doc in raw_documents:
            tokens = list(set(self.tokenizer(doc)))
            for tok in tokens:
                self.idf[tok] += 1

        for word in self.idf:
            self.idf[word] = np.log(num_docs / self.idf[word])

