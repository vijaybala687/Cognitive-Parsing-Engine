from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier on resampled/extracted EEG features.
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_svm(X_train: np.ndarray, y_train: np.ndarray, kernel: str = 'rbf', C: float = 1.0, random_state: int = 42) -> SVC:
    """
    Trains a Support Vector Classifier with an RBF kernel on EEG features.
    """
    svm_model = SVC(
        kernel=kernel,
        C=C,
        random_state=random_state
    )
    svm_model.fit(X_train, y_train)
    return svm_model



from src.models.evaluate import cross_validate_model
# Ensure your build_random_forest and build_svm functions are imported or defined above

if __name__ == "__main__":
    print("Loading real EEG features...")
    # Updated paths matching Muthuram's output
    X = np.load('data/processed/X_features.npy')
    y = np.load('data/processed/y_labels.npy')
    
    print(f"Dataset shape: {X.shape}")
    
    # Initialize empty baseline models (the cross_validator handles the .fit() step)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    
    # Run strict k-fold cross-validation
    print("\n--- Evaluating Random Forest ---")
    cross_validate_model(rf_model, X, y)
    
    print("\n--- Evaluating SVM ---")
    cross_validate_model(svm_model, X, y)