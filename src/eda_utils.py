import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column, bins=50):

    plt.figure(figsize=(12, 6))

    sns.histplot(
        df[column],
        bins=bins,
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.show()


def plot_boxplot(df, column):

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Boxplot of {column}")

    plt.show()


def missing_values_summary(df):

    missing = (
        df.isnull()
          .sum()
          .sort_values(ascending=False)
    )

    missing_percent = (
        df.isnull()
          .mean() * 100
    ).sort_values(ascending=False)

    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent
    })

    return summary