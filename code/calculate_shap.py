from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from gwrf import adaptive_bisquare_weights, fit_local_random_forest
from run_analysis import BANDWIDTH, DATA_FILE, MODEL_PARAMETERS, NON_FEATURE_COLUMNS, TARGET


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def fit_and_explain(
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
    local_features = all_features[positions]
    shap_values = shap.TreeExplainer(model).shap_values(local_features)
    return positions, np.asarray(shap_values)


def main():
    data = pd.read_csv(DATA_FILE)
    training, test = train_test_split(data, test_size=0.30, random_state=42)
    training = training.copy()
    test = test.copy()
    training["split"] = "train"
    test["split"] = "test"
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

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fit_and_explain)(
            center_index,
            training_coordinates,
            training_features,
            training_target,
            all_features,
            assigned_centers,
        )
        for center_index in range(len(training))
    )

    shap_values = np.empty_like(all_features, dtype=float)
    for positions, local_shap_values in results:
        shap_values[positions] = local_shap_values

    OUTPUT_DIR.mkdir(exist_ok=True)
    local_results = combined[["Id", "split", *features]].copy()
    for column, feature in enumerate(features):
        local_results[f"{feature}_SHAP"] = shap_values[:, column]
    local_results.to_csv(OUTPUT_DIR / "shap_values.csv", index=False)

    global_importance = pd.DataFrame(
        {
            "factor": features,
            "mean_absolute_SHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_SHAP", ascending=False)
    global_importance.to_csv(OUTPUT_DIR / "global_shap_importance.csv", index=False)


if __name__ == "__main__":
    main()
