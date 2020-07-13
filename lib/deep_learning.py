import tensorflow as tf
from lib.nlp_utils import spacy_tokenizer
import time
from collections import Counter
import numpy as np

tf.random.set_seed(999)
np.random.seed = 999

class BiLSTMLastStateClassification(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout, init=None):
        super(BiLSTMLastStateClassification, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        if init:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=init)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(enc_units))
        self.dropout = tf.keras.layers.Dropout(dropout, seed=999)
        self.fc = tf.keras.layers.Dense(enc_units, activation='relu')
        self.out = tf.keras.layers.Dense(1)
        

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x) 
        x = self.dropout(x)
        x = self.fc(x)
        output = self.out(x)

        return output
    
    
class CNNClassification(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, window, batch_sz):
        super(CNNClassification, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(enc_units, window, activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(1)
        

    def call(self, x):
        x = self.embedding(x)
        x = self.cnn(x) 
        x = self.pool(x)
        x = self.flatten(x)
        output = self.out(x)

        return output

    
class BiLSTMPoolClassification(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout, init=None):
        super(BiLSTMPoolClassification, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if init:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=init)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(enc_units, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(dropout, seed=999)
        self.fc = tf.keras.layers.Dense(enc_units, activation='relu')
        self.out = tf.keras.layers.Dense(1)
        

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x) 
        x = tf.reduce_sum(x, axis=1)
        x = self.dropout(x)
        x = self.fc(x)
        output = self.out(x)

        return output

    
def load_dataset(dataset):
    x_train, x_test, y_train, y_test = dataset
    X_train_tok = spacy_tokenizer(x_train)
    X_test_tok = spacy_tokenizer(x_test)
    
    X_train_seq, X_test_seq, word2idx = tok2seq(X_train_tok, X_test_tok)
    
    train =  tf.data.Dataset.from_generator(
        lambda: iter(zip(X_train_seq, y_train)),     
        (tf.int64, tf.int64),
        (tf.TensorShape([None]), tf.TensorShape([]))
    )
    
    test =  tf.data.Dataset.from_generator(
        lambda: iter(zip(X_test_seq, y_test)),     
        (tf.int64, tf.int64),
        (tf.TensorShape([None]), tf.TensorShape([]))
    )     
    
    return train, test, word2idx


def tok2seq(X_train_tok, X_test_tok):
    counter = Counter(w for s in X_train_tok for w in s)
    rare_words = set([k for k in counter if counter[k] == 1])
    idx2word = [word for word in counter if word not in rare_words]
    word2idx = {word:idx + 1 for idx, word in enumerate(idx2word)}
    word2idx["<<UNK>>"] = 0
        
    X_train_seq = [[word2idx.get(w, 0) for w in s] for s in X_train_tok]     
    X_test_seq = [[word2idx.get(w, 0) for w in s] for s in X_test_tok]     
    return X_train_seq, X_test_seq, word2idx
    

def run_lstm_pipeline(dataset, model, embeddings_size=64, hidden_unit=64, dropout=0.0, word2vec=None):
    train_dataset, test_dataset, word2idx = load_dataset(dataset)
    run_lstm_pipeline_(train_dataset, test_dataset, model, word2idx, embeddings_size=embeddings_size, hidden_unit=hidden_unit, dropout=dropout, word2vec=word2vec)

def create_init(word2vec, word2idx, embeddings_size):
    # https://stackoverflow.com/questions/55770009/how-to-use-a-pre-trained-embedding-matrix-in-tensorflow-2-0-rnn-as-initial-weigh
    if word2vec.vector_size != embeddings_size:
        print("Unable to Init Embeddings, the vector_size of word2vec is not equal to LSTM embeddings_size")
        return None
    
    init_np = np.random.uniform(0,1, (len(word2idx), embeddings_size))
    for word, idx in word2idx.items():
        if word in word2vec:
            init_np[idx, :] = word2vec[word]

    return tf.keras.initializers.Constant(init_np)
            
    
def run_lstm_pipeline_(train, test, model, word2idx, embeddings_size=64, hidden_unit=64, dropout=0.0, word2vec=None):
    # hyperparameters
    buffer = 10000
    batch_size = 64
        
    # dataset
    train_dataset = train.shuffle(buffer)
    test_dataset = test.shuffle(buffer)
    train_dataset = train_dataset.padded_batch(batch_size)
    test_dataset = test_dataset.padded_batch(batch_size)
    
    # model and optimizer
    if word2vec:
        init = create_init(word2vec, word2idx, embeddings_size)
    else:
        init = None
        
    vocab_size = len(word2idx)
    if model == "BiLSTMLastStateClassification":
        model = BiLSTMLastStateClassification(vocab_size, embeddings_size, hidden_unit, batch_size, dropout, init)
    elif model == "BiLSTMPoolClassification":
        model = BiLSTMPoolClassification(vocab_size, embeddings_size, hidden_unit, batch_size, dropout, init)
    else:
        print("Invalid Model Class")
        return
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # metrices
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    
    # train
    @tf.function(experimental_relax_shapes=True) # https://github.com/tensorflow/tensorflow/issues/34025
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y, tf.keras.activations.sigmoid(predictions))


    @tf.function(experimental_relax_shapes=True) # https://github.com/tensorflow/tensorflow/issues/34025
    def test_step(x, y):
        predictions = model(x, training=False)
        loss = loss_object(y, predictions)
        test_loss(loss)
        test_accuracy(y, tf.keras.activations.sigmoid(predictions))
    
    for epoch in range(10):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        start = time.time()

        for inp, targ in train_dataset:        
            train_step(inp, targ)

        for inp, targ in test_dataset:        
            test_step(inp, targ)

        template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Time: {:.2f} s'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result(),
            test_loss.result(),
            test_accuracy.result(),
            time.time() - start))
    
    
def run_cnn_pipeline(dataset, embeddings_size=64, hidden_unit=64, window=5):
    train, test, word2idx = load_dataset(dataset)
    vocab_size = len(word2idx)
    # hyperparameters
    buffer = 10000
    batch_size = 64
        
    # dataset
    train_dataset = train.shuffle(buffer)
    test_dataset = test.shuffle(buffer)
    train_dataset = train_dataset.padded_batch(batch_size)
    test_dataset = test_dataset.padded_batch(batch_size)
    
    # model and optimizer

    model = CNNClassification(vocab_size, embeddings_size, hidden_unit, window, batch_size)
 
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # metrices
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    
    # train
    @tf.function(experimental_relax_shapes=True) # https://github.com/tensorflow/tensorflow/issues/34025
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y, tf.keras.activations.sigmoid(predictions))


    @tf.function(experimental_relax_shapes=True) # https://github.com/tensorflow/tensorflow/issues/34025
    def test_step(x, y):
        predictions = model(x, training=False)
        loss = loss_object(y, predictions)
        test_loss(loss)
        test_accuracy(y, tf.keras.activations.sigmoid(predictions))
    
    for epoch in range(20):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        start = time.time()

        for inp, targ in train_dataset:        
            train_step(inp, targ)

        for inp, targ in test_dataset:        
            test_step(inp, targ)

        template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Time: {:.2f} s'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result(),
            test_loss.result(),
            test_accuracy.result(),
            time.time() - start))