import re
import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TECH_SKILLS
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
class AdvancedTextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words -= {"c", "r", "go", "js", "ai", "ml"}

    def clean_and_normalize(self, text: str) -> str:
        """Clean and normalize text with tech skill replacements."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"c\+\+", "cpp", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"\.net", "dotnet", text)
        for abbr, full in TECH_SKILLS.items():
            text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
        text = self.remove_noise(text)
        return text

    def remove_noise(self, text: str) -> str:
        """Remove URLs, emails, HTML tags, numbers, and special characters."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = re.sub(r"http\S+|www.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_words(self, tokens: list) -> list:
        """Normalize shorthand tokens to full words."""
        if not tokens or not isinstance(tokens, list):
            return []
        normalization_map = {
            "u": "you", "r": "are", "btw": "by the way",
            "js": "javascript", "py": "python",
            "ml": "machine learning", "ai": "artificial intelligence"
        }
        return [normalization_map.get(token.lower(), token) for token in tokens]

    def standardize_words(self, tokens: list) -> list:
        """Standardize variations of tech-related words."""
        if not tokens or not isinstance(tokens, list):
            return []
        standardization_map = {
            "js": "javascript", "javascript": "javascript",
            "reactjs": "react", "py": "python", "python3": "python"
        }
        return [standardization_map.get(token.lower(), token) for token in tokens]

    def preprocess_text(self, text_input):
        """Preprocess text input."""
        if isinstance(text_input, str):
            text = self.clean_and_normalize(text_input)
            tokens = [self.lemmatizer.lemmatize(w) for w in word_tokenize(text)
                      if w not in self.stop_words and len(w) > 1]
            return " ".join(tokens) if tokens else ""
        return text_input.fillna("").astype(str).apply(self.preprocess_text)
# Export standalone functions for test compatibility
def remove_noise(text: str) -> str:
    proc = AdvancedTextPreprocessor()
    return proc.remove_noise(text)

def normalize_words(tokens: list) -> list:
    proc = AdvancedTextPreprocessor()
    return proc.normalize_words(tokens)

def standardize_words(tokens: list) -> list:
    proc = AdvancedTextPreprocessor()
    return proc.standardize_words(tokens)

def preprocess_text(text_input):
    proc = AdvancedTextPreprocessor()
    return proc.preprocess_text(text_input)