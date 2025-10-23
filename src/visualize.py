"""
Visualization module for the Titanic dataset.
Reusable functions to generate key EDA plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def survival_by_gender(df: pd.DataFrame, save_path: str | Path | None = None):
    """Bar plot of survival rate by gender."""
    rates = df.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=rates.index, y=rates.values, palette="pastel", ax=ax)
    ax.set_title("Survival Rate by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rates


def survival_by_class(df: pd.DataFrame, save_path: str | Path | None = None):
    """Bar plot of survival rate by passenger class."""
    rates = df.groupby("Pclass")["Survived"].mean().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=rates.index.astype(str), y=rates.values, palette="Set2", ax=ax)
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rates


def age_distribution(df: pd.DataFrame, save_path: str | Path | None = None):
    """Histogram of ages by survival status."""
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, palette="coolwarm", ax=ax)
    ax.set_title("Age Distribution by Survival")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame, save_path: str | Path | None = None):
    """Heatmap of numeric feature correlations."""
    num = df.select_dtypes(include="number")
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
