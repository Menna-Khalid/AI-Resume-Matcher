import os,sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(''))))
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Create directories
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# Global parameters
RANDOM_STATE = 42
TARGET_ACCURACY = 0.90
# TF-IDF settings
TFIDF_PARAMS = {
    "max_features": None,  
    "ngram_range": (1, 3),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "analyzer": "word"
}

# Technical skills mapping
TECH_SKILLS = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "c++": "cpp", "c#": "csharp", "objective-c": "objc",
    "reactjs": "react", "vuejs": "vue", "nodejs": "node",
    "sklearn": "scikit-learn", "tf": "tensorflow",
    "aws": "amazon web services", "gcp": "google cloud platform",
    "ml": "machine learning", "ai": "artificial intelligence",
    "dl": "deep learning", "nlp": "natural language processing"
}