import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import preprocess_text, AdvancedTextPreprocessor
from src.models import MatchDataset, EnsembleModel
from src.feature_engineering import FeatureEngineer
from src.config import MODELS_DIR
class CVJDMatcher:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or MODELS_DIR
        self.ensemble_model = None
        self.tfidf_vectorizer = None
        self.tokenizer = None
        self.traditional_model = None
        self.bert_trainer = None
        self.is_loaded = False
        self.expected_features = None
        self.model_type = None 
    def load_models(self, model_type='ensemble'):
        """Load models with proper feature dimension handling."""
        try:
            print(f"Loading {model_type} models from {self.model_dir}...")
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(self.model_dir, 'tfidf.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                print(f"TF-IDF loaded: {self.tfidf_vectorizer.get_feature_names_out().shape[0]} features")
            else:
                raise FileNotFoundError(f"TF-IDF vectorizer not found at {tfidf_path}")
            # Load traditional model
            traditional_paths = [
                os.path.join(self.model_dir, 'best_traditional.pkl'),
                os.path.join(self.model_dir, 'best_traditional_model.pkl')
            ]
            traditional_loaded = False
            for path in traditional_paths:
                if os.path.exists(path):
                    self.traditional_model = joblib.load(path)
                    # Get expected number of features from model
                    if hasattr(self.traditional_model, 'n_features_in_'):
                        self.expected_features = self.traditional_model.n_features_in_
                    elif hasattr(self.traditional_model, 'coef_'):
                        self.expected_features = self.traditional_model.coef_.shape[1]
                    print(f"Traditional model loaded from {path}")
                    print(f"Expected features: {self.expected_features}")
                    traditional_loaded = True
                    break
            
            if not traditional_loaded:
                print("Warning: Traditional model not found")
                if model_type == 'traditional':
                    return False
            
            # Load BERT models if needed
            if model_type in ['ensemble', 'bert', 'auto']:
                bert_path = os.path.join(self.model_dir, 'bert_model')
                if os.path.exists(bert_path):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
                        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                        
                        # Create trainer for inference
                        training_args = TrainingArguments(
                            output_dir=bert_path,
                            per_device_eval_batch_size=1,
                            logging_level='error'
                        )
                        self.bert_trainer = Trainer(model=bert_model, args=training_args)
                        print("BERT model loaded successfully")
                    except Exception as e:
                        print(f"Warning: Could not load BERT model: {e}")
                        if model_type == 'bert':
                            return False
            # Create ensemble if both models available
            if model_type == 'ensemble' and self.traditional_model and self.bert_trainer:
                self.ensemble_model = EnsembleModel(
                    self.traditional_model, 
                    self.bert_trainer, 
                    self.tokenizer
                )
                print("Ensemble model created")
            
            self.model_type = model_type
            self.is_loaded = True
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False
    def _create_features(self, cv_text, jd_text):
        """Create feature matrix with proper dimensions."""
        # Preprocess texts
        processed_cv = preprocess_text(cv_text)
        processed_jd = preprocess_text(jd_text)
        combined_text = f"{processed_cv} [SEP] {processed_jd}"
        # Get TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([combined_text])
        # Create similarity features
        fe = FeatureEngineer()
        similarity_features = fe.create_similarity_features([processed_cv], [processed_jd])
        similarity_sparse = csr_matrix(similarity_features)
        # Combine features
        combined_features = hstack([tfidf_features, similarity_sparse])
        # Handle feature dimension mismatch
        if self.expected_features and combined_features.shape[1] != self.expected_features:
            current_features = combined_features.shape[1]
            if current_features > self.expected_features:
                # Truncate excess features
                combined_features = combined_features[:, :self.expected_features]
                print(f"Truncated features from {current_features} to {self.expected_features}")
            elif current_features < self.expected_features:
                # Pad with zeros
                padding_size = self.expected_features - current_features
                padding = csr_matrix((1, padding_size))
                combined_features = hstack([combined_features, padding])
                print(f"Padded features from {current_features} to {self.expected_features}")
        return combined_features, combined_text
    
    def predict(self, cv_skill, jd_requirement):
        """Make prediction with proper error handling."""
        if not self.is_loaded:
            return {'error': 'Models not loaded! Call load_models() first.'}
        try:
            # Create features
            X_features, combined_text = self._create_features(cv_skill, jd_requirement)
            print(f"Final feature matrix shape: {X_features.shape}")
            # Make prediction based on available models
            if self.ensemble_model:
                # Use ensemble
                predictions = self.ensemble_model.predict(X_features, [combined_text])
                probabilities = self.ensemble_model.predict_proba(X_features, [combined_text])
                prediction = predictions[0]
                match_probability = probabilities[0][1]
                confidence = np.max(probabilities[0])
                model_used = "Ensemble"
            elif self.traditional_model:
                # Use traditional model only
                predictions = self.traditional_model.predict(X_features)
                probabilities = self.traditional_model.predict_proba(X_features)
                prediction = predictions[0]
                match_probability = probabilities[0][1]
                confidence = np.max(probabilities[0])
                model_used = "Traditional"
            elif self.bert_trainer and self.tokenizer:
                # Use BERT only
                dataset = MatchDataset([combined_text], [0], self.tokenizer)
                bert_predictions = self.bert_trainer.predict(dataset)
                probabilities = torch.softmax(torch.from_numpy(bert_predictions.predictions), dim=-1).numpy()
                prediction = np.argmax(probabilities[0])
                match_probability = probabilities[0][1]
                confidence = np.max(probabilities[0])
                model_used = "BERT"
            else:
                return {'error': 'No valid models available'}
            
            return {
                'prediction': int(prediction),
                'match_status': 'Match' if prediction == 1 else 'No Match',
                'match_probability': float(match_probability),
                'confidence': float(confidence),
                'model_used': model_used
            }
            
        except Exception as e:
            return {
                'prediction': 0,
                'match_status': 'Error',
                'error': str(e)
            }
    
    def predict_batch(self, cv_jd_pairs):
        """Process multiple CV-JD pairs efficiently."""
        results = []
        print(f"Processing {len(cv_jd_pairs)} CV-JD pairs...")
        
        for i, (cv, jd) in enumerate(cv_jd_pairs):
            result = self.predict(cv, jd)
            result['pair_id'] = i
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(cv_jd_pairs)} pairs")
        return results

def main():
    """Test the inference system."""
    print("Testing CV-JD Matching System...")
    matcher = CVJDMatcher()
    # Try different model types
    for model_type in ['traditional', 'ensemble']:
        print(f"\nTesting with {model_type} model...")
        if matcher.load_models(model_type):
            break
        print(f"Failed to load {model_type} model, trying next...")
    else:
        print("Could not load any models!")
        return
    # Test cases
    test_cases = [
        {
            'cv': 'Senior Python developer with 8+ years experience in machine learning, deep learning frameworks like TensorFlow and PyTorch, data science, pandas, numpy, scikit-learn, REST API development, AWS cloud deployment, Docker, Kubernetes',
            'jd': 'Looking for ML Engineer with Python programming, machine learning frameworks, cloud experience, 5+ years experience in AI/ML field, TensorFlow or PyTorch required'
        },
        {
            'cv': 'Marketing manager with 3 years experience in digital marketing, social media campaigns, content strategy, brand management, Google Analytics, Facebook Ads',
            'jd': 'Senior software engineer position requiring Python, JavaScript, React, Node.js, database design, 7+ years development experience'
        }
    ]
    
    print("\nCV-JD MATCHING RESULTS")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"CV: {case['cv'][:100]}...")
        print(f"JD: {case['jd'][:100]}...")
        
        result = matcher.predict(case['cv'], case['jd'])
        
        if 'error' not in result:
            print(f"Result: {result['match_status']}")
            print(f"Probability: {result['match_probability']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Model: {result['model_used']}")
        else:
            print(f"Error: {result['error']}")
        print("-" * 60)

if __name__ == "__main__":
    main()