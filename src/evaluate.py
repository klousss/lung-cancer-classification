import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.config import RANDOM_STATE
from src.preprocessing import load_and_preprocess

def plot_roc_curves(models_info, X_test, y_test):
    plt.figure()
    for name, model in models_info.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        print('No feature importances available for this model.')
        return None

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure()
    plt.bar(fi_df['feature'], fi_df['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    return plt

def main():
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Загрузка всех сохранённых моделей
    models_dir = 'models'
    models_info = {}
    for fname in os.listdir(models_dir):
        if fname.endswith('.joblib'):
            model_name = fname.replace('.joblib', '')
            models_info[model_name] = joblib.load(os.path.join(models_dir, fname))

    # ROC-кривые
    roc_plot = plot_roc_curves(models_info, X_test, y_test)
    os.makedirs('reports/figures', exist_ok=True)
    roc_path = os.path.join('reports', 'figures', 'roc_curves.png')
    roc_plot.savefig(roc_path)
    print(f'Saved ROC plot to {roc_path}')

    # Важности признаков для лучшей модели
    best_name = max(models_info, key=lambda n: auc(*roc_curve(y_test, models_info[n].predict_proba(X_test)[:,1])[:2]))
    best_model = models_info[best_name]
    fi_plot = plot_feature_importance(best_model, X_test.columns)
    if fi_plot:
        fi_path = os.path.join('reports', 'figures', 'feature_importance.png')
        fi_plot.savefig(fi_path)
        print(f'Saved feature importance plot to {fi_path}')

if __name__ == '__main__':
    main()
