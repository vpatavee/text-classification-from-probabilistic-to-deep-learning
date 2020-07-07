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