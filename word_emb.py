from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from dense_vector_repr import DenseVectorizer
from nlp_utils import spacy_tokenizer

random_state = 999


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

def run_pipeline(dataset, embeddings, **kwargs):
    
    x_train, x_test, y_train, y_test = dataset
    
    # tokenize
    X_train_tok = spacy_tokenizer(x_train, **kwargs)
    X_test_tok = spacy_tokenizer(x_test,  **kwargs)

    # vectorize
    dense = DenseVectorizer(embeddings, **kwargs)
    x_train_vect = dense.fit_transform(X_train_tok, )
    x_test_vect = dense.transform(X_test_tok)

    # train model and evaluate
    clf = train_and_eval_model(x_train_vect, x_test_vect, y_train, y_test)
    
    return clf, dense
    