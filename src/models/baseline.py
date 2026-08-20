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


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from src.models.balancing import balance_classes
    from src.models.evaluate import evaluate_classifier

    # Generate synthetic 4-class imbalanced data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_classes=4, weights=[0.1, 0.2, 0.3, 0.4], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Balance training data
    X_train_res, y_train_res = balance_classes(X_train, y_train)

    # 2. Train baseline models
    rf = train_random_forest(X_train_res, y_train_res)
    svm = train_svm(X_train_res, y_train_res)

    # 3. Evaluate
    evaluate_classifier(rf, X_test, y_test, model_name="Random Forest")
    evaluate_classifier(svm, X_test, y_test, model_name="Support Vector Machine (RBF)")

    