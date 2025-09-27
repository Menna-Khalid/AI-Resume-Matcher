#predict.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
class PredictionRequest(BaseModel):
    cv_skill: str
    jd_requirement: str
    model_type: str = "auto"
    include_similarity: bool = True
class BatchRequest(BaseModel):
    predictions: List[PredictionRequest]
def get_model_service():
    from main import model_service
    return model_service
@router.post("/predictions/single")
async def predict_single(request: PredictionRequest, model_service = Depends(get_model_service)):
    """Single prediction"""
    try:
        result = await model_service.predict(
            cv_text=request.cv_skill,
            job_text=request.jd_requirement,
            model_type=request.model_type
        )

        return {
            'status': 'success',
            'data': result,
            'message': 'Prediction completed successfully'
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/batch")
async def predict_batch(request: BatchRequest, model_service = Depends(get_model_service)):
    """Batch predictions"""
    try:
        if len(request.predictions) > 100:
            raise HTTPException(status_code=400, detail="Max 100 predictions per batch")

        results = []
        for pred in request.predictions:
            result = await model_service.predict(
                cv_text=pred.cv_skill,
                job_text=pred.jd_requirement,
                model_type=pred.model_type
            )
            results.append(result)

        return {
            'status': 'success',
            'data': results,
            'total_count': len(request.predictions),
            'success_count': len(results),
            'message': 'Batch processing completed'
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/compare")
async def compare_models(request: PredictionRequest, model_service = Depends(get_model_service)):
    """Compare different models"""
    try:
        models = ['traditional', 'bert', 'ensemble']
        results = {}

        for model in models:
            result = await model_service.predict(
                cv_text=request.cv_skill,
                job_text=request.jd_requirement,
                model_type=model
            )
            results[model] = {
                **result,
                'available': True
            }
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['confidence'])
        return {
            'status': 'success',
            'model_results': results,
            'recommendation': f'Recommended: {best_model}',
            'message': 'Model comparison completed'
        }
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/models")
async def get_models():
    """Get available models"""
    return {
        'status': 'success',
        'available_models': [
            {
                'type': 'traditional',
                'name': 'Traditional ML',
                'description': 'TF-IDF + ML algorithms',
                'loaded': True,
                'speed': 'Fast',
                'accuracy': 'Good'
            },
            {
                'type': 'bert',
                'name': 'BERT Transformer',
                'description': 'Fine-tuned BERT model',
                'loaded': True,
                'speed': 'Medium',
                'accuracy': 'High'
            },
            {
                'type': 'ensemble',
                'name': 'Ensemble Model',
                'description': 'Combines multiple models',
                'loaded': True,
                'speed': 'Medium',
                'accuracy': 'Highest'
            }
        ]
    }
@router.get("/predictions/stats")
async def get_stats(model_service = Depends(get_model_service)):
    """Get prediction statistics"""
    try:
        stats = await model_service.get_stats()
        return {
            'status': 'success',
            'statistics': stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))