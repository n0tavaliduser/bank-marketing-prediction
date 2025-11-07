import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data_preprocessing import load_data, preprocess_data
from .model_training import train_and_evaluate

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

    for model_name in models:
        for split_ratio in split_ratios:
            for k_fold in k_folds:
                print(f'Running {model_name} with split {split_ratio} and k-fold {k_fold}')
                accuracy, mean_cv_accuracy, training_time, memory_usage = train_and_evaluate(X, y, model_name, split_ratio, k_fold, output_dir)
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