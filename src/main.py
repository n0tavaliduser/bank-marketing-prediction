import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from data_preprocessing import load_data, preprocess_data
from model_training import train_and_evaluate

def main():
    """Main function to run the pipeline."""
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets', 'bank-full.csv')
    output_dir = os.path.join(base_dir, 'output')

    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df.copy())

    X = df.drop('y', axis=1)
    y = df['y']

    # Define models, splits, and k-folds
    models = ['knn', 'decision_tree', 'naive_bayes']
    split_ratios = [0.7, 0.8, 0.9]
    k_folds = [5, 10]

    results = []
    trade_off_results = []

    for split_ratio in split_ratios:
        roc_data = {}
        for model_name in models:
            for k_fold in k_folds:
                print(f'Running {model_name} with split {split_ratio} and k-fold {k_fold}')
                accuracy, mean_cv_accuracy, training_time, memory_usage, roc_values = train_and_evaluate(X, y, model_name, split_ratio, k_fold, output_dir)
                results.append({
                    'Model': model_name,
                    'Split Ratio': split_ratio,
                    'K-Fold': k_fold,
                    'Split Accuracy': accuracy,
                    'Mean CV Accuracy': mean_cv_accuracy
                })
                trade_off_results.append({
                    'Model': model_name,
                    'Split Ratio': split_ratio,
                    'K-Fold': k_fold,
                    'Accuracy': accuracy,
                    'Training Time (s)': training_time,
                    'Memory Usage (MB)': memory_usage / (1024 * 1024)
                })
                if k_fold == 5:  # Store ROC data for one k-fold to avoid clutter
                    roc_data[model_name] = roc_values

        # Create combined ROC plot for the current split ratio
        plt.figure(figsize=(10, 8))
        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, lw=2, label=f'{model_name.upper()} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Combined ROC Curve (Split {int(split_ratio*100)}-{int((1-split_ratio)*100)})')
        plt.legend(loc="lower right")
        roc_dir = os.path.join(output_dir, 'roc-curves')
        plt.savefig(os.path.join(roc_dir, f'combined_roc_split_{int(split_ratio*100)}_{int((1-split_ratio)*100)}.png'))
        plt.close()

    # Create and save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'scorer', 'summary_results.csv'), index=False)

    # Create and save trade-off summary
    trade_off_df = pd.DataFrame(trade_off_results)
    trade_off_df.to_csv(os.path.join(output_dir, 'trade_off.csv'), index=False)

    # Create and save bar plots
    for metric in ['Split Accuracy', 'Mean CV Accuracy']:
        plt.figure(figsize=(12, 8))
        sns.barplot(data=results_df, x='Model', y=metric, hue='Split Ratio')
        plt.title(f'{metric} for all Models and Split Ratios')
        plt.savefig(os.path.join(output_dir, 'scorer', f"{metric.lower().replace(' ', '_')}_summary.png"))
        plt.close()

if __name__ == '__main__':
    main()