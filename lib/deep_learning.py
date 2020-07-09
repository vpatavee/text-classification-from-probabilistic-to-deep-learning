# from https://www.tensorflow.org/tutorials/text/text_classification_rnn

import tensorflow as tf

# This parameters should be set based on resource of your machine
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def create_lstm_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])    
    
    return model

def run_lstm_pipeline(train_dataset, test_dataset, info):
    encoder = info.features['text'].encoder
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes= ([None], []))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes= ([None], []))
    
    model = create_lstm_model(encoder.vocab_size)
    history = model.fit(
        train_dataset, epochs=10,
        validation_data=test_dataset, 
        validation_steps=30
    )
    
    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    
    
class RNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(RNN, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(enc_units))
        self.fc = tf.keras.layers.Dense(enc_units, activation='relu')
        self.out = tf.keras.layers.Dense(1)
        

    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.lstm(x, initial_state=hidden) # do you need initial_state?
        x = self.fc(x)
        output = self.out(x)
        return output

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
    
    
    