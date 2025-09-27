import os
import logging
import pandas as pd
import numpy as np
import joblib
from transformers import RobertaTokenizer, Trainer, RobertaForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util, SentenceTransformer
from preprocessing import preprocess_text
from feature_engineering import combine_features, get_sentence_embeddings
from models import train_traditional_models, train_bert, EnsembleModel, MatchDataset
from evaluate import evaluate_model
from utils import save_model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def train_pipeline(train_df, val_df, label_col="ai_prediction", save_models=True, use_saved_bert=False):
    logger.info("Starting training...")
    for df in [train_df, val_df]:
        if "cv_final" not in df:
            df["cv_final"] = preprocess_text(df["cv_skill"])
        if "jd_final" not in df:
            df["jd_final"] = preprocess_text(df["jd_requirement"])

    X_train, X_val, feats = combine_features(train_df, val_df)
    y_train, y_val = train_df[label_col].values, val_df[label_col].values
    logger.info(f"Training features shape: {X_train.shape}")
    # Traditional Models
    trad_path = os.path.join("models", "best_traditional.pkl")
    if os.path.exists(trad_path):
        logger.info("Loading saved traditional model...")
        trad_best = joblib.load(trad_path)
        trad_preds = trad_best.predict(X_val)
        trad_val_acc = evaluate_model(y_val, trad_preds)['accuracy']
        trad_res = {"best_model": "loaded", "loaded": {"model": trad_best, "val_accuracy": trad_val_acc, "predictions": trad_preds}}
    else:
        trad_res = train_traditional_models(X_train, y_train, X_val, y_val)
        trad_best = trad_res[trad_res["best_model"]]["model"]
        if save_models:
            save_model(trad_best, trad_path)
            logger.info(f"Traditional model saved with {X_train.shape[1]} features")
    # Sentence Embeddings
    sent_path = os.path.join("models", "sentence_model")
    if os.path.exists(sent_path):
        emb_model = SentenceTransformer(sent_path)
    else:
        emb_model = SentenceTransformer("all-mpnet-base-v2")
    train_texts = train_df["cv_final"] + " [SEP] " + train_df["jd_final"]
    val_texts = val_df["cv_final"] + " [SEP] " + val_df["jd_final"]
    train_emb = emb_model.encode(train_texts, show_progress_bar=True)
    val_emb = emb_model.encode(val_texts, show_progress_bar=True)
    emb_cosine = util.cos_sim(train_emb, val_emb).numpy()
    if save_models and not os.path.exists(sent_path):
        emb_model.save(sent_path)
    # BERT
    bert_path = os.path.join("models", "bert_model")
    if os.path.exists(bert_path):
        logger.info("Loading saved BERT model...")
        use_saved_bert = True
        tok = RobertaTokenizer.from_pretrained(bert_path)
        bert_model = RobertaForSequenceClassification.from_pretrained(bert_path)
        bert_trainer = Trainer(model=bert_model)
    else:
        tok = RobertaTokenizer.from_pretrained("roberta-base")
        bert_trainer = train_bert(train_texts, y_train, val_texts, y_val, tok, skip_training=use_saved_bert)
        if save_models:
            bert_trainer.save_model(bert_path)
            tok.save_pretrained(bert_path)

    val_ds = MatchDataset(val_texts, y_val, tok)
    bert_preds = np.argmax(bert_trainer.predict(val_ds).predictions, axis=1)
    bert_metrics = evaluate_model(y_val, bert_preds)
    # Ensemble
    ens = EnsembleModel(trad_best, bert_trainer, tok)
    ens_preds = ens.predict(X_val, val_texts)
    ens_metrics = evaluate_model(y_val, ens_preds)
    comparison = {"Traditional": trad_res[trad_res["best_model"]]["val_accuracy"], "BERT": bert_metrics, "Ensemble": ens_metrics}
    logger.info(f"Comparison: {comparison}")
    if save_models:
        os.makedirs("models", exist_ok=True)
        save_model(feats["tfidf"], os.path.join("models", "tfidf.pkl"))
        with open(os.path.join("models", "tfidf_vocab.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(feats["tfidf"].vocabulary_.keys()))
        logger.info(f"TF-IDF vectorizer saved with {len(feats['tfidf'].vocabulary_)} features")
    return {"traditional": trad_res, "bert": bert_metrics, "ensemble": ens_metrics}
def main():
    train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join("data", "processed", "val.csv"))
    train_pipeline(train_df, val_df, use_saved_bert=False)
if __name__ == "__main__":
    main()