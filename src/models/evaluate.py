import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def cross_validate_model(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Evaluates the model using Stratified K-Fold cross-validation to prevent data leakage.
    """
    print(f"\n--- Starting {n_splits}-Fold Cross Validation ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        fold_accuracies.append(acc)
        print(f"Fold {fold}: Accuracy = {acc:.4f}")
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"Overall Cross-Validation Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    return mean_acc, std_acc