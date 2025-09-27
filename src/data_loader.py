import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from langdetect import detect

def load_hf_dataset():
    dataset = load_dataset("lengocquangLAB/gemini-cv-skill-jd-req-match")
    return dataset["train"].to_pandas()

def filter_english(df):
    def is_english(text):
        if not isinstance(text, str) or not text.strip():
            return False
        try:
            return detect(text) == "en" and len(text.split()) > 2
        except:
            return False

    df = df[df["cv_skill"].apply(is_english) & df["jd_requirement"].apply(is_english)]
    print(f"Filtered to {len(df)} English rows")
    return df

def split_and_save(df):
    train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df["ai_prediction"])
    train, val = train_test_split(train_val, test_size=0.111, random_state=42, stratify=train_val["ai_prediction"])

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("✅ Splits saved: train/val/test")

def main():
    if os.path.exists("data/processed/train.csv"):
        print("CSVs already exist. Skipping.")
        return
    df = load_hf_dataset()
    df = filter_english(df)
    split_and_save(df)

if __name__ == "__main__":
    main()
