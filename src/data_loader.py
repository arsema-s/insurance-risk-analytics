import pandas as pd
import numpy as np
import os


def load_data(path: str) -> pd.DataFrame:

    """
    Load and preprocess the insurance dataset.

    This function:
    - Loads the raw pipe-delimited dataset
    - Cleans missing values
    - Converts numeric and datetime columns
    - Encodes binary variables
    - Engineers analytical features

    Parameters
    ----------
    path : str
        Path to the raw insurance dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with engineered features.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If the dataset cannot be loaded correctly.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    try:

        df = pd.read_csv(
            path,
            sep="|",
            low_memory=False
        )

    except Exception as e:

        raise ValueError(
            f"Error loading dataset: {e}"
        )

    # CLEAN OBJECT COLUMNS

    object_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in object_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    df.replace(
        ["", " ", "nan", "None"],
        np.nan,
        inplace=True
    )

    # DATE CONVERSION

    date_cols = [
        "TransactionMonth",
        "VehicleIntroDate"
    ]

    for col in date_cols:
        if col in df.columns:
            if "TransactionMonth" in df.columns:

                df["TransactionMonth"] = pd.to_datetime(
                    df["TransactionMonth"],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce"
    )
            if "VehicleIntroDate" in df.columns:

                df["VehicleIntroDate"] = pd.to_datetime(
                    df["VehicleIntroDate"],
                    format="%m/%Y",
                    errors="coerce"
                )

    # BOOLEAN CONVERSION

    binary_cols = [
        "AlarmImmobiliser",
        "TrackingDevice",
        "WrittenOff",
        "Rebuilt",
        "Converted",
        "CrossBorder",
        "NewVehicle",
        "IsVATRegistered"
    ]

    mapping = {
        "Yes": 1,
        "No": 0,
        "True": 1,
        "False": 0
    }

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # NUMERIC CONVERSION

    numeric_cols = [
        "TotalPremium",
        "TotalClaims",
        "CustomValueEstimate",
        "RegistrationYear",
        "kilowatts",
        "cubiccapacity",
        "NumberOfDoors",
        "Cylinders",
        "SumInsured",
        "CalculatedPremiumPerTerm"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # FEATURE ENGINEERING

    if (
        "TotalClaims" in df.columns and
        "TotalPremium" in df.columns
    ):

        df["LossRatio"] = (
            df["TotalClaims"] /
            df["TotalPremium"]
        )

        df["LossRatio"] = df[
            "LossRatio"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        df["Margin"] = (
            df["TotalPremium"] -
            df["TotalClaims"]
        )

        df["HasClaim"] = (
            df["TotalClaims"] > 0
        ).astype(int)

    if "RegistrationYear" in df.columns:

        df["VehicleAge"] = (
            2015 - df["RegistrationYear"]
        )
    return df
