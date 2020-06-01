# modified from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/word2vec.py

import tensorflow as tf
import numpy as np
import collections
import os
import random
import urllib.request
import zipfile



class MyWord2VecModel():
    def __init__(self, embedding_size, vocabulary_size, num_sampled):

        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.num_sampled = num_sampled

        initializer = tf.random_normal_initializer()

        self.embedding = tf.Variable(initial_value=initializer(shape=[vocabulary_size, embedding_size], dtype=tf.float32),trainable=True)
        self.nce_weights = tf.Variable(initial_value=initializer(shape=[vocabulary_size, embedding_size], dtype=tf.float32),trainable=True)
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), trainable=True)
        
    def __call__(self, x): # not being used during train
        X_embed = tf.nn.embedding_lookup(self.embedding, x)
        y = tf.transpose(tf.linalg.matmul(self.nce_weights, X_embed, transpose_b=True)) + self.nce_biases 
        return y 

    def loss(self, x, y):
    
        X_embed = tf.nn.embedding_lookup(self.embedding, x)
        loss_op = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=y,
            inputs=X_embed,
            num_sampled=self.num_sampled,
            num_classes=self.vocabulary_size))
        return loss_op




def load_data_set():
    url = 'http://mattmahoney.net/dc/text8.zip'
    data_path = 'text8.zip'
    if not os.path.exists(data_path):
        print("Downloading the dataset... (It may take some time)")
        filename, _ = urllib.request.urlretrieve(url, data_path)
        print("Done!")
    # Unzip the dataset file. Text has already been processed
    with zipfile.ZipFile(data_path) as f:
        text_words = f.read(f.namelist()[0]).lower().split()
    return text_words


# def build_vocab(text_words):
#     # Build the dictionary and replace rare words with UNK token
#     count = [('UNK', -1)]
#     # Retrieve the most common words
#     count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
#     # Remove samples with less than 'min_occurrence' occurrences
#     for i in range(len(count) - 1, -1, -1):
#         if count[i][1] < min_occurrence:
#             count.pop(i)
#         else:
#             # The collection is ordered, so stop when 'min_occurrence' is reached
#             break
#     # Compute the vocabulary size
#     vocabulary_size = len(count)
#     # Assign an id to each word
#     word2id = dict()
#     for i, (word, _)in enumerate(count):
#         word2id[word] = i

#     data = list()
#     unk_count = 0
#     for word in text_words:
#         # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
#         index = word2id.get(word, 0)
#         if index == 0:
#             unk_count += 1
#         data.append(index)
#     count[0] = ('UNK', unk_count)
#     id2word = dict(zip(word2id.values(), word2id.keys()))

#     print("Words count:", len(text_words))
#     print("Unique words:", len(set(text_words)))
#     print("Vocabulary size:", vocabulary_size)
#     print("Most common words:", count[:10])
#     return data, id2word

# def gen_data(data, num_skips, skip_window):
#     data_index = 0
#     # get window size (words left and right + current one)
#     span = 2 * skip_window + 1
#     buffer = collections.deque(maxlen=span) 

#     buffer.extend(data[data_index:data_index + span])
#     data_index += span
#     while data_index < len(data):
#         context_words = [w for w in range(span) if w != skip_window] 
#         words_to_use = random.sample(context_words, num_skips)
#         for context_word in words_to_use:
#             x = buffer[skip_window]
#             y = buffer[context_word]
#             yield x, y       
#         buffer.append(data[data_index])     
#         data_index += 1

# def create_date_set(data, num_skips, skip_window):
#     def gen_data():
#         data_index = 0
#         # get window size (words left and right + current one)
#         span = 2 * skip_window + 1
#         buffer = collections.deque(maxlen=span) 

#         buffer.extend(data[data_index:data_index + span])
#         data_index += span
#         while data_index < len(data):
#             context_words = [w for w in range(span) if w != skip_window] 
#             words_to_use = random.sample(context_words, num_skips)
#             for context_word in words_to_use:
#                 x = buffer[skip_window]
#                 y = buffer[context_word]
#                 yield x, [y]      
#             buffer.append(data[data_index])     
#             data_index += 1

#     dataset = tf.data.Dataset.from_generator(
#         gen_data,
#         (tf.int64, tf.int64),
#         (tf.TensorShape([]), tf.TensorShape([1])),        
#     )

#     return dataset


class MyWord2Vec:
    def __init__(self, corpus, embedding_size, max_vocabulary_size, min_occurrence, skip_window, batch_size, epoch, num_skips=2, num_sampled=64):
        # Word2Vec Parameters
        self.embedding_size = embedding_size # Dimension of the embedding vector
        self.max_vocabulary_size = max_vocabulary_size # Total number of different words in the vocabulary
        self.min_occurrence = min_occurrence # Remove all words that does not appears at least n times
        self.skip_window = skip_window # How many words to consider left and right
        self.num_skips = num_skips # How many times to reuse an input to generate a label
        self.num_sampled = 64 # Number of negative examples to sample
        self.batch_size = batch_size
        self.corpus = corpus
        self.data, self.id2word = self.build_vocab(self.corpus)
        self.model = MyWord2VecModel(
            embedding_size=self.embedding_size, 
            vocabulary_size=len(self.id2word), 
            num_sampled=self.num_sampled
            )
        self.dataset = self.create_date_set(self.data).batch(self.batch_size)
        self.epoch = epoch
        self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

    def build_vocab(self, text_words):
        # Build the dictionary and replace rare words with UNK token
        count = [('UNK', -1)]
        # Retrieve the most common words
        count.extend(collections.Counter(text_words).most_common(self.max_vocabulary_size - 1))
        # Remove samples with less than 'min_occurrence' occurrences
        for i in range(len(count) - 1, -1, -1):
            if count[i][1] < self.min_occurrence:
                count.pop(i)
            else:
                # The collection is ordered, so stop when 'min_occurrence' is reached
                break
        # Compute the vocabulary size
        vocabulary_size = len(count)
        # Assign an id to each word
        word2id = dict()
        for i, (word, _)in enumerate(count):
            word2id[word] = i

        data = list()
        unk_count = 0
        for word in text_words:
            # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
            index = word2id.get(word, 0)
            if index == 0:
                unk_count += 1
            data.append(index)
        count[0] = ('UNK', unk_count)
        id2word = dict(zip(word2id.values(), word2id.keys()))

        print("Words count:", len(text_words))
        print("Unique words:", len(set(text_words)))
        print("Vocabulary size:", vocabulary_size)
        print("Most common words:", count[:10])
        return data, id2word

    def train(self):
        for e in range(self.epoch):
            loss, n = 0, 0
            for x, y in self.dataset:
                loss += self.train_step(x,y)
                n +=1
            print("epoch {} loss {}".format(e, loss.numpy()/n))

        emb = self.model.embedding
        return emb, self.id2word

    def create_date_set(self, data):
        def gen_data():
            data_index = 0
            # get window size (words left and right + current one)
            span = 2 * self.skip_window + 1
            buffer = collections.deque(maxlen=span) 

            buffer.extend(data[data_index:data_index + span])
            data_index += span
            while data_index < len(data):
                context_words = [w for w in range(span) if w != self.skip_window] 
                words_to_use = random.sample(context_words, self.num_skips)
                for context_word in words_to_use:
                    x = buffer[self.skip_window]
                    y = buffer[context_word]
                    yield x, [y]      
                buffer.append(data[data_index])     
                data_index += 1

        dataset = tf.data.Dataset.from_generator(
            gen_data,
            (tf.int64, tf.int64),
            (tf.TensorShape([]), tf.TensorShape([1])),        
        )

        return dataset

    @tf.function
    def train_step(self, x ,y):
        loss = self.model.loss(x, y)
        self.opt.minimize(
            loss, 
            var_list=[self.model.embedding, self.model.nce_weights, self.model.nce_biases])
        return loss

    

# create toy data set
# x = tf.convert_to_tensor(np.array([3,2,4,3]), dtype='int64')
# label =  tf.convert_to_tensor(np.array([5,4,1,2]).reshape(-1,1), dtype='int64')
# print(x)
# print(label)

# create dataset
# text_words = load_data_set()

# dataset = create_date_set(data,2, 3).batch(4)

# create model
# model = MyWord2VecModel(embedding_size=10, vocabulary_size=len(id2word), num_sampled=2)

# # optimizer
# opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

# train
# @tf.function
# def train_step(x ,y):
#     loss = model.loss(x, y)

#     opt.minimize(loss, var_list=[model.embedding, model.nce_weights, model.nce_biases])
#     return loss

# # train_step()
# for epoch in range(50):
#     loss, n = 0, 0
#     for i, (x,y) in enumerate(dataset):
#         # print("batch", i)
#         loss += train_step(x,y)
#         n +=1
#     print(loss.numpy()/n)

# emb = model.embedding
# print(emb)



# print(model.loss(x, label))

if __name__ == "__main__":
    corpus = load_data_set()
    print(corpus[:10])
    embedding_size = 300
    max_vocabulary_size = 100000
    min_occurrence = 2
    skip_window = 10
    batch_size = 100
    epoch = 10


    myWV = MyWord2Vec(corpus, embedding_size, max_vocabulary_size, min_occurrence, skip_window, batch_size, epoch, num_skips=2, num_sampled=64)
    emb = myWV.train()
    print(emb.shape)
