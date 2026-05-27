import pandas as pd

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


def province_claim_ttest(
    df,
    province_a,
    province_b
):

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