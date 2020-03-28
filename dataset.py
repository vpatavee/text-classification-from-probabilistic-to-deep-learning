import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import pickle

PICKLE_FILE_NAME = "imdb_data.pkl"

def download_tfds_imdb_as_text():
    """
    @return 
    X_train: list of string
    y_train: list of int
    X_test: list of string
    y_test: list of int
    """
    
    (train_data, test_data), _ = tfds.load(
        'imdb_reviews/plain_text', 
        split = (tfds.Split.TRAIN, tfds.Split.TEST), 
        with_info=True, as_supervised=True) 
    
    X_train = [e[0].numpy().decode("utf-8") for e in train_data ]
    X_test = [e[0].numpy().decode("utf-8") for e in test_data ]
    
    y_train = [e[1].numpy() for e in train_data ]
    y_test = [e[1].numpy() for e in test_data ]
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    print("number of training samples", len(X_train))
    print("number of testing samples", len(X_test))
    
    return X_train, X_test, y_train, y_test

def download_tfds_imdb_as_tensor_subword_8k():
    (train_data, test_data), info = tfds.load(
        'imdb_reviews/subwords8k', 
        split = (tfds.Split.TRAIN, tfds.Split.TEST), 
        with_info=True, as_supervised=True)

    return train_data, test_data, info
