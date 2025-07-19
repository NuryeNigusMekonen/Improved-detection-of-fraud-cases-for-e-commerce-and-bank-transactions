import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

def plot_fraud_rate_by_category(df, category_col, target_col='class', save=False):
    top_cats = df[category_col].value_counts().head(10).index
    df_top = df[df[category_col].isin(top_cats)]
    prop_df = pd.crosstab(df_top[category_col], df_top[target_col], normalize='index')
    prop_df.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(10,5))
    plt.title(f'Fraud Rate by Top {category_col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(f"../reports/figures/eda/fraud_rate_{category_col}.png")
    plt.close()
