import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column, bins=50):

    """
    Plot the distribution of a numeric feature using a histogram and KDE.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    column : str
        Column to visualize.

    bins : int, optional
        Number of histogram bins, by default 50.

    Returns
    -------
    None
    """

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

    """
    Plot a boxplot for detecting outliers in a numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    column : str
        Column to visualize.

    Returns
    -------
    None
    """

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Boxplot of {column}")

    plt.show()


def missing_values_summary(df):

    """
    Generate a summary table of missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Table containing missing value counts and percentages.
    """

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