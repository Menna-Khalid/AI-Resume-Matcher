import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(''))))
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import joblib,os
from src.config import TFIDF_PARAMS, MODELS_DIR
from src.preprocessing import AdvancedTextPreprocessor

class FeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        self.preproc = AdvancedTextPreprocessor()

    def create_similarity_features(self, cvs, jds):
        feats = []
        for cv, jd in zip(cvs, jds):
            cv_words, jd_words = set(cv.split()), set(jd.split())
            inter, union = len(cv_words & jd_words), len(cv_words | jd_words)
            jaccard = inter / union if union else 0
            feats.append([jaccard, len(cv_words), len(jd_words)])
        return np.array(feats)

def get_tfidf_features(train_texts, val_texts):
    fe = FeatureEngineer()
    X_train = fe.tfidf.fit_transform(train_texts)
    X_val = fe.tfidf.transform(val_texts)
    joblib.dump(fe.tfidf, os.path.join(MODELS_DIR, "tfidf.pkl"))
    return X_train, X_val, fe.tfidf

def get_sentence_embeddings(train_texts, val_texts, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    train_emb = model.encode(train_texts.tolist(), batch_size=32, show_progress_bar=True)
    val_emb = model.encode(val_texts.tolist(), batch_size=32, show_progress_bar=True)
    model.save(os.path.join(MODELS_DIR, "sentence_model"))
    return train_emb, val_emb, model

def combine_features(train_df, val_df, cv_col="cv_final", jd_col="jd_final"):
    train_comb, val_comb = train_df[cv_col] + " [SEP] " + train_df[jd_col], val_df[cv_col] + " [SEP] " + val_df[jd_col]
    train_tfidf, val_tfidf, tfidf = get_tfidf_features(train_comb, val_comb)
    fe = FeatureEngineer()
    sim_train = fe.create_similarity_features(train_df[cv_col], train_df[jd_col])
    sim_val = fe.create_similarity_features(val_df[cv_col], val_df[jd_col])
    X_train = hstack([train_tfidf, csr_matrix(sim_train)])
    X_val = hstack([val_tfidf, csr_matrix(sim_val)])
    return X_train, X_val, {"tfidf": tfidf, "feature_engineer": fe}