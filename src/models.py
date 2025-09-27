import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

class MatchDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_traditional_models(X_train, y_train, X_val, y_val):
    models = {
        "naive_bayes": MultinomialNB(),
        "log_reg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
        "gb": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "svm": SVC(probability=True, class_weight="balanced", kernel="rbf")
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        results[name] = {
            "model": model,
            "val_accuracy": accuracy_score(y_val, val_preds)
        }
    best_model = max(results, key=lambda x: results[x]["val_accuracy"])
    results["best_model"] = best_model
    results["best_accuracy"] = results[best_model]["val_accuracy"]
    return results

def train_bert(train_texts, train_labels, val_texts, val_labels, tokenizer,
               model_name="roberta-base", output_dir="models/bert_results", skip_training=False):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_ds = MatchDataset(train_texts, train_labels, tokenizer)
    val_ds = MatchDataset(val_texts, val_labels, tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=16,  
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01 
    )
    def compute_metrics(pred):
        return {"accuracy": accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=-1))}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds if not skip_training else None,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    if not skip_training:
        trainer.train()
    return trainer

class EnsembleModel:
    def __init__(self, ml_model, bert_trainer, tokenizer, weights=(0.3, 0.7)):
        self.ml_model = ml_model
        self.bert_trainer = bert_trainer
        self.tokenizer = tokenizer
        self.weights = weights

    def predict(self, X_ml, texts):
        ml_proba = self.ml_model.predict_proba(X_ml)
        bert_ds = MatchDataset(texts, [0] * len(texts), self.tokenizer)
        bert_preds = self.bert_trainer.predict(bert_ds)
        bert_proba = torch.softmax(torch.from_numpy(bert_preds.predictions), dim=-1).numpy()
        combined_proba = self.weights[0] * ml_proba + self.weights[1] * bert_proba
        return np.argmax(combined_proba, axis=1)