# utils.py
import pandas as pd
import numpy as np
import joblib
import json
from collections import Counter
from typing import List, Dict, Any
from nltk.util import ngrams
def save_model(model, filepath: str):
    """Save model to file"""
    try:
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_model(filepath: str):
    """Load model from file"""
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def analyze_text_statistics(text_series: pd.Series, title: str = "Text Analysis"):
    """Analyze text statistics"""
    lengths = text_series.str.len()
    word_counts = text_series.str.split().str.len()
    
    stats = {
        'total_texts': len(text_series),
        'avg_length': lengths.mean(),
        'median_length': lengths.median(),
        'avg_words': word_counts.mean(),
        'median_words': word_counts.median()
    }
    
    print(f"\n{title}:")
    print(f"Total texts: {stats['total_texts']}")
    print(f"Avg length: {stats['avg_length']:.1f} chars")
    print(f"Avg words: {stats['avg_words']:.1f}")
    return stats

def get_word_frequency(text_series: pd.Series, top_k: int = 15):
    """Get most frequent words"""
    all_words = []
    for text in text_series.dropna():
        words = str(text).lower().split()
        all_words.extend([w for w in words if len(w) > 2 and w.isalpha()])
    freq = Counter(all_words).most_common(top_k)
    print(f"\nTop {top_k} words:")
    for word, count in freq:
        print(f"{word}: {count}")
    return freq

def analyze_class_distribution(df: pd.DataFrame, target_col: str = 'ai_prediction'):
    """Analyze target variable distribution"""
    dist = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    print(f"\nClass Distribution:")
    for class_val in sorted(dist.index):
        count = dist[class_val]
        pct = percentages[class_val]
        print(f"Class {class_val}: {count} ({pct:.1f}%)")
    # Check balance
    ratio = min(percentages) / max(percentages)
    if ratio < 0.4:
        print(f"Warning: Dataset is imbalanced (ratio: {ratio:.2f})")
    else:
        print("Dataset is reasonably balanced")
    return dist

def save_training_results(results: Dict[str, Any], filepath: str = "training_results.json"):
    """Save training results to JSON"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Clean results for JSON
    clean_results = {}
    for key, value in results.items():
        if key not in ['predictions', 'confusion_matrix']:  # Skip non-serializable
            clean_results[key] = convert_numpy(value)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def get_ngrams(text_series, n=2, top_k=10):
    all_ngrams = []
    for text in text_series.dropna():
        tokens = str(text).lower().split()
        all_ngrams.extend(ngrams(tokens, n))
    freq = Counter(all_ngrams).most_common(top_k)
    return freq

def count_stopwords(text_series, stop_words):
    count = 0
    for text in text_series.dropna():
        words = str(text).lower().split()
        count += sum(1 for w in words if w in stop_words)
    return count

def stopword_ratio(text_series, stop_words):
    total_words = 0
    stop_count = 0
    for text in text_series.dropna():
        words = str(text).lower().split()
        total_words += len(words)
        stop_count += sum(1 for w in words if w in stop_words)
    return stop_count / total_words if total_words > 0 else 0

def create_performance_summary(results: Dict[str, Any]):
    """Create performance summary"""
    print("\nPERFORMANCE SUMMARY")
    print("=" * 40)
    if 'best_model' in results:
        print(f"Best Model: {results['best_model']}")
    if 'best_f1' in results:
        print(f"Best F1 Score: {results['best_f1']:.4f}")
    if 'training_time' in results:
        print(f"Training Time: {results['training_time']:.1f} minutes")
    print("=" * 40)