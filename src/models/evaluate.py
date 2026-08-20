from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Map targets to human-readable motor imagery classes
CLASS_NAMES = [
    "Left-hand imagery",
    "Right-hand imagery",
    "Both-hands imagery",
    "Both-feet imagery"
]

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Classifier"):
    """
    Computes and prints standard classification metrics and confusion matrix.
    """
    predictions = model.predict(X_test)
    
    print(f"\n==================== {model_name} Evaluation ====================")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("=" * 60)