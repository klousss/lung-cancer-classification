import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_STATE
from src.preprocessing import load_and_preprocess


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    models = {
        'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'lda': LinearDiscriminantAnalysis(),
        'svm': SVC(probability=True, random_state=RANDOM_STATE),
        'decision_tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }

    best_model = None
    best_score = 0.0
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        results[name] = score
        print(f"{name}: ROC-AUC = {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\nЛучший алгоритм: {best_name} с ROC-AUC = {best_score:.4f}")

    # Сохранение лучшей модели и результатов
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f"{best_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"Сохранена лучшая модель: {model_path}")

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['roc_auc'])
    results_path = os.path.join('models', 'roc_auc_results.csv')
    results_df.to_csv(results_path)
    print(f"Сохранены результаты ROC-AUC: {results_path}")


if __name__ == '__main__':
    main()