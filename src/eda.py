import os
import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths for saving figures
UNIVARIANT_DIR = "../reports/figures/univariant"
BIVARIANT_DIR = "../reports/figures/bivariant"

os.makedirs(UNIVARIANT_DIR, exist_ok=True)
os.makedirs(BIVARIANT_DIR, exist_ok=True)


def save_figure(fig: plt.Figure, filename: Optional[str], folder: str) -> None:
    """
    Saves the matplotlib figure to the specified folder with given filename.
    """
    if filename:
        path = os.path.join(folder, filename)
        fig.savefig(path, dpi=300)
        logging.info(f"Saved figure: {path}")



def plot_transaction_amount_distribution(df: pd.DataFrame, amount_col: str = 'purchase_value',
                                         title: str = "Transaction Amount Distribution",
                                         save: bool = False, filename: Optional[str] = None) -> plt.Figure:
    """
    Plots the histogram and KDE of the transaction amounts.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[amount_col].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    if save:
        save_figure(fig, filename, UNIVARIANT_DIR)

    return fig



def plot_correlation_heatmap(df, title="Feature Correlation Heatmap", save=False, filename=None):
    """
    Plots heatmap of the correlation matrix of dataframe numeric features only.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select numeric columns only to avoid conversion errors
    numeric_df = df.select_dtypes(include=['number'])
    
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    fig.tight_layout()

    if save and filename:
        fig.savefig(f"../reports/figures/bivariant/{filename}", dpi=300)

    return fig



def plot_amount_vs_class(df, amount_col='purchase_value', target_col='class', title="Amount vs Fraud Class", save=False, filename=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Cast x-axis to categorical explicitly if needed:
    df[target_col] = df[target_col].astype('category')
    
    # Remove palette or adjust as per warning:
    sns.boxplot(data=df, x=target_col, y=amount_col, ax=ax)  # Removed palette
    
    ax.set_title(title)
    ax.set_xlabel("Class (0 = Legit, 1 = Fraud)")
    ax.set_ylabel("Amount")
    fig.tight_layout()

    if save and filename:
        fig.savefig(f"../reports/figures/bivariant/{filename}", dpi=300)

    return fig



def plot_time_distribution(df: pd.DataFrame, time_col: str = 'signup_time',
                           title: str = "Signup Time Distribution",
                           save: bool = False, filename: Optional[str] = None) -> plt.Figure:
    """
    Plots histogram of hour-of-day for a datetime column.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['hour'] = df[time_col].dt.hour
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['hour'].dropna(), bins=24, kde=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    fig.tight_layout()

    if save:
        save_figure(fig, filename, UNIVARIANT_DIR)

    return fig


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str], save: bool = False) -> List[plt.Figure]:
    """
    Plots histograms with KDE for multiple numeric columns.
    """
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        if save:
            filename = f"{col}_distribution.png"
            save_figure(fig, filename, UNIVARIANT_DIR)

        figs.append(fig)
    return figs


def plot_class_distribution(df: pd.DataFrame, target_col: str = 'class',
                            title: str = "Class Distribution",
                            save: bool = False, filename: Optional[str] = None) -> plt.Figure:
    """
    Plots the distribution of classes (binary classification).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Ensure target_col is categorical to avoid warnings
    if not pd.api.types.is_categorical_dtype(df[target_col]):
        df[target_col] = df[target_col].astype('category')
        logging.info(f"Converted '{target_col}' to categorical dtype for plotting.")

    # Use palette without hue (deprecated) --> instead use color or default
    sns.countplot(data=df, x=target_col, color='steelblue', ax=ax)  # simpler color fix

    ax.set_title(title)
    ax.set_xlabel("Class (0 = Legit, 1 = Fraud)")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    fig.tight_layout()

    if save:
        save_figure(fig, filename, UNIVARIANT_DIR)

    return fig


def plot_categorical_feature_distribution(df: pd.DataFrame, column: str,
                                          title: Optional[str] = None,
                                          save: bool = False, filename: Optional[str] = None) -> plt.Figure:
    """
    Plots countplot of a categorical feature.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Convert to categorical if not already
    if not pd.api.types.is_categorical_dtype(df[column]):
        df[column] = df[column].astype('category')
        logging.info(f"Converted '{column}' to categorical dtype for plotting.")

    sns.countplot(data=df, x=column, color='steelblue', ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    plt.xticks(rotation=45)
    fig.tight_layout()

    if save:
        save_figure(fig, filename, UNIVARIANT_DIR)

    return fig


def plot_categorical_by_target(df: pd.DataFrame, column: str, target_col: str = 'class',
                               save: bool = False) -> plt.Figure:
    """
    Plots stacked bar chart of proportion of target variable per category level.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Ensure both columns categorical
    for col in [column, target_col]:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
            logging.info(f"Converted '{col}' to categorical dtype for plotting.")

    prop_df = pd.crosstab(df[column], df[target_col], normalize='index')
    prop_df.plot(kind='bar', stacked=True, colormap="coolwarm", ax=ax)
    ax.set_title(f"Fraud Rate by {column}")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(column)
    plt.xticks(rotation=45)
    fig.tight_layout()

    if save:
        filename = f"{column}_fraud_rate.png"
        save_figure(fig, filename, BIVARIANT_DIR)

    return fig

