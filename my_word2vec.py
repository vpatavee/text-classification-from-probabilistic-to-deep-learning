import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import itertools
tf.keras.backend.set_floatx('float64')


corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
# convert to lower case
corpus_raw = corpus_raw.lower()


class MyWord2VecModel(Model):
    def __init__(self, dim, vocab_size):
        super(MyWord2VecModel, self).__init__()
        self.w1 = Dense(dim)
        self.w2 = Dense(vocab_size)

    def call(self, x):
        """
        @param x one hot encoding of shape = batch_size x vocab_size
        @return probability distribution of prediction word
        """
        hidden_representation = self.w1(x)  # shape = batch_size x dim
        logit = self.w2(hidden_representation) # shape = batch_size x vocab_size
        pred = tf.nn.softmax(logit) # shape = batch_size x vocab_size
        return pred

class MyWord2Vec():
    def __init__(self, dim, window_size, epoch, batch_size, buffer_size, ignore=None):

        self.dim = dim
        self.vocab_size = None
        self.window_size = window_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.word2int = dict()
        self.int2word = list()
        self.ignore = set(ignore) if ignore else set()

    def train(self, corpus):
        dataset = self.process_corpus(corpus)
        model = MyWord2VecModel(self.dim, self.vocab_size)
        model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        model.fit(x=dataset,epochs=self.epoch)

        w1, b1 = model.w1.get_weights()
        w2, b2 = model.w2.get_weights()
        return ((w1 + b1) + (w2 + b2).T) /2 , self.int2word
        

    def process_corpus(self, corpus_raw):
        """
        @corpus_raw  list of list of tokens, like gensim
        return training instance
        for example (window size=2)
        corpus_raw 
            [['he', 'is', 'the', 'king'], ['the', 'king', 'is', 'royal'], ['she', 'is', 'the', 'royal', 'queen']]
        return 
            [['he', 'is'], ['he', 'the'], ['is', 'he'], ['is', 'the'], ['is', 'king'], ['the', 'he'], 
            ['the', 'is'], ['the', 'king'], ['king', 'is'], ['king', 'the'], ['the', 'king'], ['the', 'is'], 
            ['king', 'the'], ['king', 'is'], ['king', 'royal'], ['is', 'the'], ['is', 'king'], ['is', 'royal'], 
            ['royal', 'king'], ['royal', 'is'], ['she', 'is'], ['she', 'the'], ['is', 'she'], ['is', 'the'], 
            ['is', 'royal'], ['the', 'she'], ['the', 'is'], ['the', 'royal'], ['the', 'queen'], ['royal', 'is'], 
            ['royal', 'the'], ['royal', 'queen'], ['queen', 'the'], ['queen', 'royal']]
        """
        words = set()

        for word in list(itertools.chain(*corpus_raw)):
            if word not in self.ignore: 
                words.add(word)

        self.vocab_size = len(words) 
        self.int2word = list(set(words))
        self.vocab_size = len(self.int2word)
        self.word2int = {word:i for i, word in enumerate(words)}

        data = list()
        for sentence in corpus_raw:
            for word_index, word in enumerate(sentence):
                for nb_word in sentence[max(word_index - self.window_size, 0) : min(word_index + self.window_size, len(sentence)) + 1] : 
                    if nb_word != word and word not in self.ignore and nb_word not in self.ignore:
                        data.append([word, nb_word])

        x_train = np.zeros([len(data), self.vocab_size])
        y_train = np.zeros([len(data), self.vocab_size])

        print(x_train.shape)

        for i, data_word in enumerate(data):
            x_train[i, self.word2int[data_word[0]]] = 1
            y_train[i, self.word2int[data_word[1]]] = 1

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(self.buffer_size).batch(self.buffer_size)

        return dataset
