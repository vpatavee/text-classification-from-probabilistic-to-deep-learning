from dense_vector_repr import DenseVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
from nlp_utils import preprocess_remove_html_non_ascii, preprocess_remove_html, spacy_tokenizer, spacy_tokenizer_lower, \
    spacy_tokenizer_lower_lemma, spacy_tokenizer_lower_lemma_remove_stop, spacy_tokenizer_lower_lemma_remove_stop_and_punc


def create_dense_vects(preprocessors, tokenizers, pollings, tfidf, models):

    dense_vects = dict()

    for preprocessor in preprocessors:
        for p in pollings:
            for tf in tfidf:
                for tokenizer in tokenizers:
                    for model_name, model in models.items():

                        dense_vect = DenseVectorizer(
                            model=model,
                            tfidf=tf,
                            preprocessor=preprocess_remove_html_non_ascii,
                            tokenizer=tokenizer,
                            polling=p,
                        )
                        dense_vects[(
                            preprocessor.__name__,
                            p,
                            tf,
                            tokenizer.__name__,
                            model_name
                        )] = dense_vect

    return dense_vects


def run_logistic_word_emb_exp(X_train, X_test, y_train, y_test, models):

    # models = {"word2vec": model_word2vec, "glove": model_glove}
    preprocessors = [preprocess_remove_html_non_ascii]
    tfidf = [True, False]
    tokenizers = [spacy_tokenizer_lower_lemma_remove_stop, spacy_tokenizer, spacy_tokenizer_lower_lemma]
    pollings = [
        DenseVectorizer.Polling.norm,
        DenseVectorizer.Polling.log,
        DenseVectorizer.Polling.sum
    ]

    dense_vects = create_dense_vects(
        preprocessors, tokenizers, pollings, tfidf, models)
    res = dict()

    for name, dense_vect in dense_vects.items():

        X_train_vec = dense_vect.fit_transform(X_train)
        X_test_vec = dense_vect.transform(X_test)

        best_f1 = 0
        for c in range(-5, 5):
            C = 10 ** c
            clf = LogisticRegression(random_state=0, max_iter=1500, C=C)
            clf.fit(X_train_vec, y_train)
            y_test_pred = clf.predict(X_test_vec)
            f1 = f1_score(y_test, y_test_pred)
            best_f1 = max(f1, best_f1)

        res[name] = f1

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index()
    res_df.columns = ["index", "F1"]
    # res_df["preprocessing"] = res_df["index"].apply(lambda x: x[0])
    res_df["polling"] = res_df["index"].apply(lambda x: x[1])
    res_df["tfidf"] = res_df["index"].apply(lambda x: x[2])
    res_df["tokenizer"] = res_df["index"].apply(lambda x: x[3])
    res_df["word_emb_model"] = res_df["index"].apply(lambda x: x[4])

    return res_df[["word_emb_model", "tfidf", "tokenizer", "polling", "F1"]]
