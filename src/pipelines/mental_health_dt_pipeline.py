import re
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipelines.dt_pipeline import DTPipeline
from src.pipelines.dynamic_column_transformer import DynamicColumnTransformer


class MentalHealthDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame):
        super().__init__(X)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        # Encoding for categorical data
        categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            # ('scale', StandardScaler())
        ], memory=None)

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', categorical_encoder)
        ], memory=None)

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, self.numerical_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])

        # Bundle preprocessing
        return Pipeline(steps=[
            # standard preprocessing for remaining missing data
            ('preprocessor', preprocessor)
        ], memory=None)
