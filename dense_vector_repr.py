import gensim
from collections import defaultdict, Counter
import numpy as np
from numpy import linalg
from nlp_utils import preprocess_remove_html_non_ascii, preprocess_remove_html, spacy_tokenizer, spacy_tokenizer_lower, \
    spacy_tokenizer_lower_lemma, spacy_tokenizer_lower_lemma_remove_stop, spacy_tokenizer_lower_lemma_remove_stop_and_punc


# Try to make it resemble to Scikit-learn Countvectorizer
class DenseVectorizer:
    class Polling:
        norm = "norm"
        sum = "SUM"
        log = "LOG"
        
    def __init__(self, model, tokenizer, preprocessor, tfidf, polling, is_extend_unk_word=False, print_stat=False):
        self.model = model 
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.tfidf = tfidf
        self.polling = polling
        self.is_extend_unk_word = is_extend_unk_word
        self.idx2oov = list()
        self.oov2idx = dict()
        self.print_stat = print_stat
        
        if self.tfidf:
            self.idf = defaultdict(int)
        else:
            self.idf = defaultdict(lambda: 1)

    def fit_transform(self, raw_documents):
        if self.tfidf:
            self._cal_idf(raw_documents)
        if self.is_extend_unk_word:
            self._create_oov(raw_documents)
        return self.transform(raw_documents, True)

    def transform(self, raw_documents, fit=False):
        if self.polling == self.Polling.sum:
            return self._transform_sum_norm(raw_documents, False)
        elif self.polling == self.Polling.norm:
            return self._transform_sum_norm(raw_documents, True)
        elif self.polling == self.Polling.log:
            return self._transform_log(raw_documents)

    def _transform_sum_norm(self, raw_documents, is_norm):
        num_tok = 0
        num_oov = 0 # not in model 
        num_no_found = 0 # not in model and not in self.oov2idx

        dim = self.model.vector_size + len(self.idx2oov) if self.is_extend_unk_word else self.model.vector_size 
        num_doc = len(raw_documents)
        # mat = np.zeros([num_doc, dim])

        mat = np.random.rand(num_doc, dim) * 1e-10

        for i, doc in enumerate(raw_documents):
            doc = self.preprocessor(doc)
            for token in self.tokenizer(doc):
                num_tok += 1
                if self.is_extend_unk_word:
                    if token in self.model:
                        mat[i, :self.model.vector_size] += self.model[token] * self.idf[token]
                    elif token in self.oov2idx:
                        mat[i, self.oov2idx[token]] += 1.0 * self.idf[token]
                        num_oov += 1
                    else:
                        num_no_found += 1
                else:
                    if token in self.model:
                        mat[i, :] += self.model[token] * self.idf[token]
                    else:
                        num_no_found += 1
                        num_oov += 1

        if self.print_stat:
            print("# tokens", num_tok)
            print("# not in model ", num_oov)
            print("# not in model and not in self.oov2idx", num_no_found)

        if is_norm:
            return mat / linalg.norm(mat, axis=1).reshape(-1,1)
        else:
            return mat

    def _transform_log(self, raw_documents):
        num_tok = 0
        num_oov = 0 # not in model 
        num_no_found = 0 # not in model and not in self.oov2idx

        dim = self.model.vector_size + len(self.idx2oov) if self.is_extend_unk_word else self.model.vector_size
        num_doc = len(raw_documents)
        mat = np.zeros([num_doc, dim])
        # mat = np.random.rand(num_doc, dim) * 1e-10

        for i, doc in enumerate(raw_documents):
            doc = self.preprocessor(doc)
            tf = Counter(self.tokenizer(doc))
            for w, n in tf.items():
                num_tok += n
                if self.is_extend_unk_word:
                    if w in self.model:
                        mat[i, :self.model.vector_size] += (1 + np.log(n)) * self.model[w] * self.idf[w]
                    elif w in self.oov2idx:
                        mat[i, self.oov2idx[w]] += (1 + np.log(n)) * self.idf[w]
                        num_oov += n
                    else:
                        num_no_found += n
                else:
                    if w in self.model:
                        mat[i, :] += (1 + np.log(n)) * self.model[w] * self.idf[w]
                    else:
                        num_oov += n
                        num_no_found += n

        if self.print_stat:
            print("# tokens", num_tok)
            print("# not in model ", num_oov)
            print("# not in model and not in self.oov2idx", num_no_found)

        return mat

    def _cal_idf(self, raw_documents):

        num_docs = len(raw_documents)
        for doc in raw_documents:
            tokens = list(set(self.tokenizer(doc)))
            for tok in tokens:
                self.idf[tok] += 1

        for word in self.idf:
            self.idf[word] = np.log(num_docs / self.idf[word])
            
    def _create_oov(self, raw_documents):
        vocab = set()
        for doc in raw_documents:
            doc = self.preprocessor(doc)
            for token in self.tokenizer(doc):
                if token not in self.model:
                    vocab.add(token)
        
        self.idx2oov = list(vocab)
        self.oov2idx = {w:i for i, w in enumerate(self.idx2oov)}
