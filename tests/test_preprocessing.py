import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_text, remove_noise, normalize_words, standardize_words
class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions"""
    def setUp(self):
        """Set up test data"""
        self.sample_text = pd.Series([
            "Python programming with JS and React.js",
            "Machine Learning & AI development",
            "Email: test@example.com Visit: https://example.com",
            "Data Science with Python3 and ML"
        ])
    def test_remove_noise(self):
        """Test noise removal"""
        text = "Visit https://example.com or email test@example.com! <h1>Title</h1> 123 items."
        cleaned = remove_noise(text)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("<h1>", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertNotIn("!", cleaned)

    def test_normalize_words(self):
        """Test normalization"""
        tokens = ["u", "r", "btw", "js", "py", "ml", "ai"]
        normalized = normalize_words(tokens)
        expected = ["you", "are", "by the way", "javascript", "python", "machine learning", "artificial intelligence"]
        self.assertEqual(normalized, expected)

    def test_standardize_words(self):
        """Test standardization"""
        tokens = ["js", "javascript", "reactjs", "py", "python3"]
        standardized = standardize_words(tokens)
        expected = ["javascript", "javascript", "react", "python", "python"]
        self.assertEqual(standardized, expected)

    def test_preprocess_text(self):
        """Test full preprocessing"""
        processed = preprocess_text(self.sample_text)
        
        self.assertIsInstance(processed, pd.Series)
        for text in processed:
            self.assertTrue(text.islower())
            self.assertIsInstance(text, str)

    def test_preprocess_empty_text(self):
        """Test preprocessing with empty values"""
        empty_series = pd.Series([None, "", "  ", "Valid text"])
        processed = preprocess_text(empty_series)
        self.assertEqual(len(processed), 4)
        self.assertTrue(all(isinstance(text, str) for text in processed))
if __name__ == '__main__':
    unittest.main()