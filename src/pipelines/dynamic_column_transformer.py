from sklearn.compose import ColumnTransformer


class DynamicColumnTransformer(ColumnTransformer):

    def __init__(
            self,
            transformers,
            numerical_transformer,
            categorical_transformer,
            *,
            remainder="drop",
            sparse_threshold=0.3,
            n_jobs=None,
            transformer_weights=None,
            verbose=False,
            verbose_feature_names_out=True,
            force_int_remainder_cols=True,
    ):
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer
        self.transformers = transformers
        super().__init__(
            transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
            force_int_remainder_cols=force_int_remainder_cols
        )

    def set_transformers(self, X):
        numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
        categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

        # Recreate transformers with updated columns
        self.transformers = [
            ('num', self.numerical_transformer, numerical_cols),
            ('cat', self.categorical_transformer, categorical_cols)
        ]

    def fit(self, X, y=None, **params):
        self.set_transformers(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **params):
        self.set_transformers(X)
        return super().fit_transform(X, y, **params)

    def transform(self, X, **params):
        self.set_transformers(X)
        return super().transform(X, **params)
