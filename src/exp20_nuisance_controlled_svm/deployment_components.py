import numpy as np

class NuisanceResidualizer:

    def __init__(self):
        self.beta_ = None
        self.n_features_in_ = None
        self.nuisance_columns_ = None

    def fit(self, X, nuisance):

        X = np.asarray(
            X,
            dtype=float
        )

        Z = np.asarray(
            nuisance,
            dtype=float
        )

        if X.shape[0] != Z.shape[0]:
            raise ValueError(
                "X and nuisance row counts do not match."
            )

        intercept = np.ones(
            (Z.shape[0], 1),
            dtype=float
        )

        design = np.hstack(
            [
                intercept,
                Z
            ]
        )

        self.beta_ = np.linalg.lstsq(
            design,
            X,
            rcond=None
        )[0]

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X, nuisance):

        X = np.asarray(
            X,
            dtype=float
        )

        Z = np.asarray(
            nuisance,
            dtype=float
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"received {X.shape[1]}."
            )

        if X.shape[0] != Z.shape[0]:
            raise ValueError(
                "X and nuisance row counts do not match."
            )

        intercept = np.ones(
            (Z.shape[0], 1),
            dtype=float
        )

        design = np.hstack(
            [
                intercept,
                Z
            ]
        )

        return X - design @ self.beta_


# ================================================================
# FIT ONE MODEL
# ================================================================

