from pathlib import Path
import numpy as np
import joblib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mne.decoding import CSP

# ---------------------------------
# Project paths
# ---------------------------------
project_root = Path(__file__).resolve().parent.parent

processed_path = project_root / "processed"
models_path = project_root / "models"

models_path.mkdir(exist_ok=True)

# ---------------------------------
# Load processed dataset
# ---------------------------------
print("Loading processed dataset...")

X = np.load(processed_path / "X.npy")
y = np.load(processed_path / "y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------------------
# Train/Test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ---------------------------------
# Apply CSP
# ---------------------------------
print("\nApplying CSP...")

csp = CSP(
    n_components=4,
    log=True,
    norm_trace=False
)

X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp = csp.transform(X_test)

# ---------------------------------
# Train LDA
# ---------------------------------
print("Training LDA classifier...")

lda = LinearDiscriminantAnalysis()

lda.fit(X_train_csp, y_train)

# ---------------------------------
# Evaluate
# ---------------------------------
y_pred = lda.predict(X_test_csp)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))

# ---------------------------------
# Save model
# ---------------------------------
joblib.dump(lda, models_path / "lda_model.pkl")
joblib.dump(csp, models_path / "csp_model.pkl")

print("\nModels saved successfully!")

print("LDA :", models_path / "lda_model.pkl")
print("CSP :", models_path / "csp_model.pkl")