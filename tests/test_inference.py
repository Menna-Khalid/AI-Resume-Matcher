import unittest
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import CVJDMatcher

class TestInference(unittest.TestCase):
    """Test inference functionality"""
    def setUp(self):
        """Set up test matcher with real models directory"""
        self.matcher = CVJDMatcher(model_dir='models')
    def test_matcher_initialization(self):
        """Test matcher initialization"""
        self.assertIsNotNone(self.matcher)
        self.assertIsNone(self.matcher.model_type)
        logger.info("Test matcher_initialization passed")
    def test_load_models(self):
        """Test loading traditional models"""
        self.assertTrue(self.matcher.load_models('traditional'))
        self.assertIsNotNone(self.matcher.traditional_model)
        logger.info("Test load_models passed")
    def test_predict_with_model(self):
        """Test prediction with loaded traditional model"""
        if not self.matcher.load_models('traditional'):
            self.skipTest("Failed to load traditional models")
        result = self.matcher.predict("Python programming", "Software engineer with Python")
        self.assertNotIn('error', result)
        self.assertIn('prediction', result)
        self.assertIn('match_status', result)
        logger.info("Test predict_with_model passed")
if __name__ == '__main__':
    unittest.main()