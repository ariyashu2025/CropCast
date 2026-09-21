import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def generate_eda_plots(file_path='dataset/crop_yield.csv'):
    os.makedirs('static/plots', exist_ok=True)
    df = pd.read_csv(file_path).dropna()

    target_col = 'Yield' if 'Yield' in df.columns else df.columns[-1]

    # 1. Yield Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, color='teal')
    plt.title('Crop Yield Distribution')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.savefig('static/plots/yield_distribution.png')
    plt.close()

    # 2. Correlation Heatmap (for numerical columns)
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('static/plots/correlation_heatmap.png')
    plt.close()

    print("EDA plots successfully generated and saved to static/plots/!")


if __name__ == "__main__":
    generate_eda_plots()