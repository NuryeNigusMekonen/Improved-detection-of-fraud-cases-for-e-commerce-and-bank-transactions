from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kmeans_clustering(df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df['cluster'] = clusters
    return df, kmeans

def plot_clusters(df, x_col, y_col):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='cluster', palette='Set2')
    plt.title(f'Cluster Analysis: {x_col} vs {y_col}')
    plt.show()
