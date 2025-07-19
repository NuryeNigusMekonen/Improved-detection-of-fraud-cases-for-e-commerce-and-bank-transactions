import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_n_categories(df, column, top_n=10, target_col='class'):
    top_cats = df[column].value_counts().head(top_n).index
    df_top = df[df[column].isin(top_cats)]
    plt.figure(figsize=(10,5))
    sns.countplot(data=df_top, x=column, hue=target_col, palette='coolwarm')
    plt.title(f'{target_col} Distribution for Top {top_n} {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_grouped_bar_by_fraud(df, column, target_col='class', top_n=10):
    top_cats = df[column].value_counts().head(top_n).index
    df_top = df[df[column].isin(top_cats)]
    prop_df = pd.crosstab(df_top[column], df_top[target_col], normalize='index')
    prop_df.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(10,5))
    plt.title(f'Fraud Rate by Top {top_n} {column}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
