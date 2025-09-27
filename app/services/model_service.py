import logging,os,sys
import time
from typing import Dict, Any
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.models import MatchDataset
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class ModelService:
    """CV-Job matching service using only pre-trained saved models."""

    def __init__(self):
        self.initialized = False
        self.match_threshold = 0.25  # Lower threshold for matching
        self.stats = {
            'total_predictions': 0,
            'success_rate': 0.8178,
            'avg_response_time': 0.0,
            'active_models': 0
        }
        self.models = {}

    def _update_active_models(self):
        """Update the active_models count based on loaded models."""
        active_count = sum(1 for model in ['tfidf', 'traditional', 'sentence_transformer', 'bert_trainer']
                          if self.models.get(model) is not None)
        self.stats['active_models'] = active_count
        logger.info(f"Updated active models count to {active_count}")

    async def initialize(self):
        """Initialize all pre-trained models from saved files."""
        try:
            logger.info("Initializing pre-trained models...")
            # Load saved models
            tfidf_path = 'models/tfidf.pkl'
            trad_path = 'models/best_traditional.pkl'
            self.models['tfidf'] = joblib.load(tfidf_path)
            self.models['traditional'] = joblib.load(trad_path)

            # Load sentence transformer
            self.models['sentence_transformer'] = SentenceTransformer('models/sentence_model')

            # Load BERT models
            bert_path = 'models/bert_model'
            self.models['bert_tokenizer'] = AutoTokenizer.from_pretrained(bert_path)
            self.models['bert_model'] = AutoModelForSequenceClassification.from_pretrained(bert_path)
            training_args = TrainingArguments(output_dir=bert_path, per_device_eval_batch_size=1)
            self.models['bert_trainer'] = Trainer(model=self.models['bert_model'], args=training_args)

            # Update stats with initial success rate
            self.stats['success_rate'] = 0.8178  # Based on provided evaluation
            self.initialized = True
            self._update_active_models()
            logger.info("Pre-trained models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self._update_active_models()
            raise

    async def predict(self, cv_text: str, job_text: str, model_type: str = "auto") -> Dict[str, Any]:
        """Predict using pre-trained models with emphasis on similarity scores."""
        if not self.initialized:
            raise Exception("Models not initialized")
        start_time = time.time()
        combined_text = f"{cv_text} [SEP] {job_text}"
        # Compute similarity scores
        cv_tfidf = self.models['tfidf'].transform([cv_text])
        jd_tfidf = self.models['tfidf'].transform([job_text])
        tfidf_sim = cosine_similarity(cv_tfidf, jd_tfidf)[0][0]
        cv_emb = self.models['sentence_transformer'].encode(cv_text, convert_to_tensor=True)
        jd_emb = self.models['sentence_transformer'].encode(job_text, convert_to_tensor=True)
        semantic_sim = util.cos_sim(cv_emb, jd_emb).item()
        # Use pre-trained models for prediction
        if model_type.lower() == "traditional":
            features = self.models['tfidf'].transform([combined_text])
            expected_features = self.models['traditional'].n_features_in_  # 25806
            current_features = features.shape[1]
            if current_features < expected_features:
                padding_size = expected_features - current_features
                padded_features = hstack([features, csr_matrix(np.zeros((features.shape[0], padding_size)))])
            else:
                padded_features = features
            proba = self.models['traditional'].predict_proba(padded_features)[0][1]
            used_model = "traditional"
        elif model_type.lower() in ["auto", "bert"]:
            dataset = MatchDataset([combined_text], [1], self.models['bert_tokenizer'])
            predictions = self.models['bert_trainer'].predict(dataset)
            bert_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[0][1].item()
            logger.info(f"Raw BERT probability: {bert_proba}")
            similarity_weight = 0.5 * (tfidf_sim + semantic_sim) / 2
            proba = 0.5 * bert_proba + similarity_weight
            proba = max(0.0, min(proba, 1.0))
            used_model = "bert"
        elif model_type.lower() == "ensemble":
            dataset = MatchDataset([combined_text], [1], self.models['bert_tokenizer'])
            bert_predictions = self.models['bert_trainer'].predict(dataset)
            bert_proba = torch.softmax(torch.tensor(bert_predictions.predictions), dim=1)[0][1].item()
            features = self.models['tfidf'].transform([combined_text])
            expected_features = self.models['traditional'].n_features_in_  # 25806
            current_features = features.shape[1]
            if current_features < expected_features:
                padding_size = expected_features - current_features
                padded_features = hstack([features, csr_matrix(np.zeros((features.shape[0], padding_size)))])
            else:
                padded_features = features
            trad_proba = self.models['traditional'].predict_proba(padded_features)[0][1]
            combined_proba = 0.4 * trad_proba + 0.6 * bert_proba
            similarity_weight = 0.5 * (tfidf_sim + semantic_sim) / 2
            proba = 0.5 * combined_proba + similarity_weight
            used_model = "ensemble"
        else:
            raise ValueError("Invalid model_type")

        # Boost score based on similarity
        if tfidf_sim > 0.3 and semantic_sim > 0.6:
            proba = min(proba * 1.15, 1.0)
        elif tfidf_sim > 0.4 or semantic_sim > 0.7:
            proba = min(proba * 1.08, 1.0)

        proba = max(0.0, min(proba, 1.0))
        prediction = 1 if proba > self.match_threshold else 0
        confidence = proba

        processing_time = (time.time() - start_time) * 1000
        self.stats['total_predictions'] += 1
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['total_predictions'] - 1) + processing_time) 
            / self.stats['total_predictions']
        )
        return {
            'prediction': prediction,
            'probability': round(proba, 4),
            'confidence': round(confidence, 4),
            'match_status': 'Match' if prediction == 1 else 'No Match',
            'confidence_level': self._get_confidence_level(confidence),
            'model_type': used_model,
            'similarity_scores': {
                'tfidf_similarity': round(tfidf_sim, 3),
                'semantic_similarity': round(semantic_sim, 3)
            },
            'processing_time_ms': round(processing_time, 1)
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Map confidence score to level."""
        if confidence >= 0.75: return "Very High"
        elif confidence >= 0.55: return "High"
        elif confidence >= 0.35: return "Medium"
        elif confidence >= 0.2: return "Low"
        else: return "Very Low"

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.stats

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'models_loaded': self.initialized,
            'uptime': time.time()
        }