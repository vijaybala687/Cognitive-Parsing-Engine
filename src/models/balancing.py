from imblearn.over_sampling import SMOTE
import numpy as np

def balance_classes(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE to balance the 4-class motor imagery training samples.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled