"""Fit GW-RF and report training and test performance."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gwrf import adaptive_bisquare_weights, fit_local_random_forest


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "model_input.csv"
TARGET = "SUHII"
NON_FEATURE_COLUMNS = {"Longitude", "Latitude", "Id", TARGET}
BANDWIDTH = 183
MODEL_PARAMETERS = {
    "n_estimators": 367,
    "max_features": 8,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
}


def fit_and_predict(
    center_index,
    training_coordinates,
    training_features,
    training_target,
    all_features,
    assigned_centers,
):
    weights = adaptive_bisquare_weights(
        training_coordinates,
        training_coordinates[center_index],
        BANDWIDTH,
    )
    model = fit_local_random_forest(
        training_features,
        training_target,
        weights,
        MODEL_PARAMETERS,
    )
    positions = np.flatnonzero(assigned_centers == center_index)
    return positions, model.predict(all_features[positions])


def print_metrics(name, observed, predicted):
    r2 = r2_score(observed, predicted)
    rmse = mean_squared_error(observed, predicted) ** 0.5
    mae = mean_absolute_error(observed, predicted)
    print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")


def main():
    data = pd.read_csv(DATA_FILE)
    if data.isna().any().any() or data["Id"].duplicated().any():
        raise ValueError("The model input contains missing values or duplicate city IDs.")

    training, test = train_test_split(data, test_size=0.30, random_state=42)
    combined = pd.concat([training, test], ignore_index=True)
    features = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]

    training_features = training[features].to_numpy(dtype=float)
    training_target = training[TARGET].to_numpy(dtype=float)
    training_coordinates = training[["Longitude", "Latitude"]].to_numpy(dtype=float)
    test_coordinates = test[["Longitude", "Latitude"]].to_numpy(dtype=float)
    all_features = combined[features].to_numpy(dtype=float)

    nearest_training_center = np.argmin(
        cdist(test_coordinates, training_coordinates), axis=1
    )
    assigned_centers = np.concatenate(
        [np.arange(len(training)), nearest_training_center]
    )

    print(f"Fitting {len(training)} local GW-RF models...")
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fit_and_predict)(
            center_index,
            training_coordinates,
            training_features,
            training_target,
            all_features,
            assigned_centers,
        )
        for center_index in range(len(training))
    )

    predictions = np.empty(len(combined), dtype=float)
    for positions, local_predictions in results:
        predictions[positions] = local_predictions

    number_of_training_cities = len(training)
    print_metrics(
        "Training set",
        training[TARGET].to_numpy(),
        predictions[:number_of_training_cities],
    )
    print_metrics(
        "Test set",
        test[TARGET].to_numpy(),
        predictions[number_of_training_cities:],
    )


if __name__ == "__main__":
    main()
