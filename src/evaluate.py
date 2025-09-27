# evaluate.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import TARGET_ACCURACY
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """Enhanced model evaluation with comprehensive metrics"""
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'balanced_accuracy': balanced_acc,
        'predictions': y_pred,
        'confusion_matrix': cm
    }
    # Print enhanced evaluation
    print(f"\n{model_name} - ENHANCED EVALUATION")
    print("=" * 50)
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"Balanced Acc:  {balanced_acc:.4f}")

    # Target achievement
    if accuracy >= TARGET_ACCURACY:
        print(f"🎯 TARGET ACHIEVED: {accuracy*100:.1f}% >= {TARGET_ACCURACY*100:.0f}%")
    else:
        gap = (TARGET_ACCURACY - accuracy) * 100
        print(f"📊 Gap to target: {gap:.1f}% (Current: {accuracy*100:.1f}%, Target: {TARGET_ACCURACY*100:.0f}%)")
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           No Match  Match")
    print(f"No Match      {tn:4d}    {fp:4d}")
    print(f"Match         {fn:4d}    {tp:4d}")
    # Error analysis
    total = len(y_true)
    errors = fp + fn
    print(f"\nError Analysis:")
    print(f"Total samples: {total}")
    print(f"Correct: {total-errors} ({(total-errors)/total*100:.1f}%)")
    print(f"False Positives: {fp} ({fp/total*100:.1f}%)")
    print(f"False Negatives: {fn} ({fn/total*100:.1f}%)")
    # Performance insights
    if precision < 0.85:
        print("⚠️  Consider reducing false positives")
    if recall < 0.85:
        print("⚠️  Consider reducing false negatives")
    if f1 >= 0.90:
        print("✅ Excellent F1 performance!")
    return metrics

def compare_models(model_results):
    """Enhanced model comparison with ranking"""
    print(f"\n{'='*60}")
    print("ENHANCED MODEL COMPARISON")
    print(f"{'='*60}")
    # Headers
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 60)
    # Sort by F1 score
    sorted_results = sorted(
        model_results.items(), 
        key=lambda x: x[1]['f1'], 
        reverse=True
    )
    best_model = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        acc = metrics['accuracy']
        f1 = metrics['f1']
        prec = metrics['precision']
        rec = metrics['recall']
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{marker} {name:<17} {acc:<10.4f} {f1:<8.4f} {prec:<10.4f} {rec:<8.4f}")
    print("-" * 60)
    print(f"🏆 WINNER: {best_model}")
    print(f"   F1 Score: {best_metrics['f1']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    # Achievement status
    if best_metrics['accuracy'] >= TARGET_ACCURACY:
        print(f"🎉 PROJECT SUCCESS: {best_metrics['accuracy']*100:.1f}% accuracy achieved!")
    else:
        print(f"📈 Progress: {best_metrics['accuracy']*100:.1f}% (Target: {TARGET_ACCURACY*100:.0f}%)")
    return best_model, best_metrics['f1']
def quick_evaluation(model, X_test, y_test, model_name="Model"):
    """Quick evaluation wrapper for any sklearn model"""
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    return evaluate_model(y_test, y_pred, y_proba, model_name)