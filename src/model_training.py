from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def train_and_evaluate(X, y, model_name, split_ratio, k_fold, output_dir):
    """Trains and evaluates a model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42, stratify=y)

    if model_name == 'knn':
        param_grid = {'n_neighbors': range(1, 21)}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=k_fold)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
    elif model_name == 'naive_bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Create output directories
    split_dir = os.path.join(output_dir, f'scorer/splits_{int(split_ratio*100)}_{int((1-split_ratio)*100)}')
    cm_dir = os.path.join(output_dir, 'confusion-matrix')
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name.upper()} (Split {int(split_ratio*100)}-{int((1-split_ratio)*100)})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(cm_dir, f'{model_name}_split_{int(split_ratio*100)}_{int((1-split_ratio)*100)}.png'))
    plt.close()

    # Save performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    performance_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })
    performance_df.to_csv(os.path.join(split_dir, f'{model_name}_performance.csv'), index=False)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_fold)
    mean_cv_accuracy = np.mean(cv_scores)

    return accuracy, mean_cv_accuracy