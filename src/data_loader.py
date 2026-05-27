import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep="|",
        low_memory=False
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
