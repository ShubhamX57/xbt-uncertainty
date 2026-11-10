"""Select the labelled casts, build features, and split train and test by cruise."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from xbt_uncertainty.labels import add_label_columns

NUMERIC_FEATURES = ["year", "month", "lat", "lon", "max_depth"]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OTHER = "OTHER"


def labelled_ship_casts(casts):
    """Labelled, ship-launched casts that belong to a cruise."""
    casts = add_label_columns(casts)
    keep = casts["labelled"] & ~casts["is_axbt"] & casts["has_cruise"]
    return casts[keep].reset_index(drop=True)


def merge_rare_models(table, min_cruises):
    """Replace models seen on fewer than min_cruises cruises with OTHER."""
    cruises = table.groupby("model")["cruise"].nunique()
    rare = cruises.index[cruises < min_cruises]
    out = table.copy()
    out["target"] = out["model"].where(~out["model"].isin(rare), OTHER)
    return out, sorted(rare)


def split_by_cruise(table, test_fraction=0.2, seed=0):
    """Boolean mask of test rows. Every cruise falls wholly in train or in test.

    StratifiedGroupKFold keeps groups together while trying to keep the class
    mix of each fold close to the whole; one fold is taken as the test set.
    """
    n_splits = round(1 / test_fraction)
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_index = next(folds.split(table, table["target"], groups=table["cruise"]))
    is_test = np.zeros(len(table), dtype=bool)
    is_test[test_index] = True
    return is_test


def feature_encoder():
    """Country is encoded as an integer code; numbers pass through unchanged."""
    return ColumnTransformer(
        [
            (
                "country",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


class YearMajority:
    """Predict the most common class of the training casts from the same year.

    Years not seen in training fall back to the overall most common class.
    """

    def fit(self, X, y):
        frame = pd.DataFrame({"year": X["year"].to_numpy(), "y": np.asarray(y)})
        self.by_year_ = frame.groupby("year")["y"].agg(lambda s: s.value_counts().index[0])
        self.overall_ = frame["y"].value_counts().index[0]
        return self

    def predict(self, X):
        return X["year"].map(self.by_year_).fillna(self.overall_).to_numpy()
