from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipelines.dt_pipeline import DTPipeline

sleep_dictionary = {
    '1-2 hours': 1.5,
    '1-3 hours': 2,
    '2-3 hours': 2.5,
    'Unhealthy': 3,  # probably wrong
    '3-4 hours': 3.5,
    '3-6 hours': 3.5,
    '1-6 hours': 4,
    'Less than 5 hours': 4,
    '4-5 hours': 4.5,
    '4-6 hours': 5,
    '5-6 hours': 5.5,
    'Moderate': 6,  # probably wrong
    'Work_Study_Hours': 6,  # probably wrong
    '6-7 hours': 6.5,
    '6-8 hours': 7,
    '9-5': 7,
    '9-5 hours': 7,
    '7-8 hours': 7.5,
    '9-6 hours': 7.5,
    '8 hours': 8,
    '10-6 hours': 8,
    '8-9 hours': 8.5,
    "More than 8 hours": 9,
    '9-11 hours': 10,
    '10-11 hours': 10.5,
}

diet_dictionary = {
    'More Healty': 0,
    'Healthy': 1,
    'Less than Healthy': 2,
    'Less Healthy': 2,
    'Moderate': 3,
    'Unhealthy': 4,
    'No Healthy': 4,
}

gender_dictionary = {
    'Male': 0,
    'Female': 1,
}

student_profession_dictionary = {
    'Working Professional': 1,
    'Student': 0,
}

yes_no_dictionary = {
    'No': 0,
    'Yes': 1,
}


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # fill metric numericals with values that make sense (eg: if working, then no study stress)
        X['Academic_Pressure'] = X['Academic_Pressure'].fillna(0)
        X['Work_Pressure'] = X['Work_Pressure'].fillna(0)
        X['Study_Satisfaction'] = X['Study_Satisfaction'].fillna(0)
        X['Job_Satisfaction'] = X['Job_Satisfaction'].fillna(0)
        X['CGPA'] = X['CGPA'].fillna(-1)  # we want to emphasize that it's a missing value

        # fill unknown categoricals
        X['Profession'] = X['Profession'].fillna('Unknown')
        X['Dietary_Habits'] = X['Dietary_Habits'].fillna('Unknown')
        X['Degree'] = X['Degree'].fillna('Unknown')

        # Assign values based on the dictionaries and set -1 for empty values
        X['Sleep_Duration'] = X['Sleep_Duration'].apply(
            lambda x: sleep_dictionary.get(x, -1) if x != '' else -1)
        # X['Profession'] = X['Profession'].apply(lambda x: .get(x, -1) if x != '' else -1)
        X['Dietary_Habits'] = X['Dietary_Habits'].apply(lambda x: diet_dictionary.get(x, -1) if x != '' else -1)
        # X['Degree'] = X['Degree'].apply(lambda x: .get(x, -1) if x != '' else -1)
        X['Gender'] = X['Gender'].apply(lambda x: gender_dictionary.get(x, -1) if x != '' else -1)
        X['Have_you_ever_had_suicidal_thoughts_'] = X['Have_you_ever_had_suicidal_thoughts_'].apply(
            lambda x: yes_no_dictionary.get(x, -1) if x != '' else -1
        )
        X['Working_Professional_or_Student'] = X['Working_Professional_or_Student'].apply(
            lambda x: student_profession_dictionary.get(x, -1) if x != '' else -1
        )
        X['Family_History_of_Mental_Illness'] = X['Family_History_of_Mental_Illness'].apply(
            lambda x: yes_no_dictionary.get(x, -1) if x != '' else -1
        )

        return X


class MentalHealthDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame, imputation_enabled: bool):
        super().__init__(X, imputation_enabled)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        # Encoding for categorical data
        categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # if imputation is disabled, just encode categorical columns
        if not self.imputation_enabled:
            return ColumnTransformer(transformers=[
                ('cat', categorical_encoder, self.categorical_cols)
            ])

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

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
            ('custom_imputer', CustomImputer()),
            ('preprocessor', preprocessor)
        ], memory=None)
