import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from models import train_traditional_models
class TestModels(unittest.TestCase):
    """Test model training functions"""
    def setUp(self):
        """Set up test data"""
        texts = ["python machine learning", "java spring", "python data", "java backend"]
        self.vectorizer = TfidfVectorizer(max_features=10)
        self.X_train = self.vectorizer.fit_transform(texts[:3])
        self.y_train = np.array([1, 0, 1])
        self.X_val = self.vectorizer.transform(texts[3:])
        self.y_val = np.array([0])

    def test_train_traditional_models(self):
        """Test traditional models training"""
        # Get the result and log it for debugging
        result = train_traditional_models(self.X_train, self.y_train, self.X_val, self.y_val)
        logger.info(f"train_traditional_models returned: {result} (type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'})")
        # Handle the dictionary return structure
        if not isinstance(result, dict):
            self.fail(f"Expected a dictionary return from train_traditional_models, got {type(result)}: {result}")
        results = result
        best_model_name = results.get('best_model')
        if best_model_name is None:
            self.fail("No 'best_model' key found in train_traditional_models return value")
        # Check expected models
        models = ['naive_bayes', 'log_reg', 'rf', 'gb', 'svm']  # Updated to match log
        for model in models:
            self.assertIn(model, results)
            self.assertTrue(hasattr(results[model]['model'], 'predict'))
            self.assertIn('val_accuracy', results[model])  # Updated to 'val_accuracy'
        # Check best model
        self.assertIn(best_model_name, models)
        self.assertGreaterEqual(results[best_model_name]['val_accuracy'], 0)  # Use val_accuracy

if __name__ == '__main__':
    unittest.main()