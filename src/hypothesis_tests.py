import pandas as pd

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


def province_claim_ttest(
    df,
    province_a,
    province_b
):
    
    """
    Perform Welch's t-test to compare claim severity
    between two provinces.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    province_a : str
        First province to compare.

    province_b : str
        Second province to compare.

    Returns
    -------
    dict
        Dictionary containing:
        - t_statistic
        - p_value
    """

    group_a = df[
        df["Province"] == province_a
    ]["TotalClaims"].dropna()

    group_b = df[
        df["Province"] == province_b
    ]["TotalClaims"].dropna()

    statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    return {
        "t_statistic": statistic,
        "p_value": p_value
    }


def gender_claim_chi2(df):

    """
    Perform a chi-squared test to evaluate the association
    between gender and claim occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    Returns
    -------
    dict
        Dictionary containing:
        - chi2 statistic
        - p_value
    """

    contingency = pd.crosstab(
        df["Gender"],
        df["HasClaim"]
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(contingency)
    )

    return {
        "chi2": chi2,
        "p_value": p_value
    }

def zipcode_claim_ttest(
    df,
    zipcode_a,
    zipcode_b
):
    
    """
    Perform Welch's t-test to compare claim severity
    between two postal codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    zipcode_a : int
        First postal code.

    zipcode_b : int
        Second postal code.

    Returns
    -------
    dict
        Dictionary containing:
        - t_statistic
        - p_value
    """

    group_a = df[
        df["PostalCode"] == zipcode_a
    ]["TotalClaims"].dropna()

    group_b = df[
        df["PostalCode"] == zipcode_b
    ]["TotalClaims"].dropna()

    statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    return {
        "t_statistic": statistic,
        "p_value": p_value
    }

def zipcode_margin_ttest(
    df,
    zipcode_a,
    zipcode_b
):
    
    """
    Perform Welch's t-test to compare underwriting
    margin differences between two postal codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    zipcode_a : int
        First postal code.

    zipcode_b : int
        Second postal code.

    Returns
    -------
    dict
        Dictionary containing:
        - t_statistic
        - p_value
    """

    group_a = df[
        df["PostalCode"] == zipcode_a
    ]["Margin"].dropna()

    group_b = df[
        df["PostalCode"] == zipcode_b
    ]["Margin"].dropna()

    statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    return {
        "t_statistic": statistic,
        "p_value": p_value
    }