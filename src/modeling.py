from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


def prepare_model_data(df):

    """
    Prepare features, target variable, and preprocessing
    pipeline for machine learning models.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    Returns
    -------
    tuple
        Contains:
        - X : feature matrix
        - y : target variable
        - preprocessor : preprocessing pipeline
    """

    features = [
        "Province",
        "VehicleType",
        "make",
        "Model",
        "VehicleAge",
        "SumInsured",
        "kilowatts"
    ]

    target = "TotalClaims"

    X = df[features]

    y = df[target]

    categorical_features = [
        "Province",
        "VehicleType",
        "make",
        "Model"
    ]

    numerical_features = [
        "VehicleAge",
        "SumInsured",
        "kilowatts"
    ]

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                categorical_features
            ),
            (
                "num",
                numerical_transformer,
                numerical_features
            )
        ]
    )

    return X, y, preprocessor


def train_random_forest(df):

    """
    Train and evaluate a Random Forest regression model
    for insurance claim prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    Returns
    -------
    dict
        Dictionary containing:
        - trained model
        - RMSE
        - R² score
    """

    X, y, preprocessor = (
        prepare_model_data(df)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mse = mean_squared_error(
    y_test,
    predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2
    }

def train_xgboost(df):

    """
    Train and evaluate an XGBoost regression model
    for insurance claim prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    Returns
    -------
    dict
        Dictionary containing:
        - trained model
        - RMSE
        - R² score
    """

    X, y, preprocessor = (
        prepare_model_data(df)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mse = mean_squared_error(
    y_test,
    predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2
    }

def train_linear_regression(df):

    """
    Train and evaluate a Linear Regression baseline model
    for insurance claim prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input insurance dataframe.

    Returns
    -------
    dict
        Dictionary containing:
        - trained model
        - RMSE
        - R² score
    """

    X, y, preprocessor = (
        prepare_model_data(df)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2
    }