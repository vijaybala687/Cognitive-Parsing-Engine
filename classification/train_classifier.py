from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from preprocessing.apply_csp import get_csp_features

# Get features and labels
X, y = get_csp_features()

# Create classifier
clf = LinearDiscriminantAnalysis()

# 5-fold cross validation
scores = cross_val_score(clf, X, y, cv=5)

print("Accuracy for each fold:", scores)
print("Average Accuracy:", scores.mean())