import re

from pandas import DataFrame

from src.preprocessors.data_preprocessor import DataPreprocessor

sleep_dictionary = {
    '0': 0,
    'No': 0,
    '1-2 hours': 1.5,
    '1-3 hours': 2,
    '2-3 hours': 2.5,
    '20-21 hours': 2.5,  # probably per week
    'Unhealthy': 3,
    '3-4 hours': 3.5,
    '3-6 hours': 3.5,
    '1-6 hours': 4,
    'Less than 5 hours': 4,
    '4-5 hours': 4.5,
    '4-6 hours': 5,
    '35-36 hours': 5,  # probably per week
    '5-6 hours': 5.5,
    'Moderate': 6,
    '45': 6,  # probably per week
    '40-45 hours': 6,  # probably per week
    '6-7 hours': 6.5,
    '45-48 hours': 6.5,  # probably per week
    '6-8 hours': 7,
    '9-5': 7,
    '9-5 hours': 7,
    '49 hours': 7,  # probably per week
    '7-8 hours': 7.5,
    '9-6 hours': 7.5,
    '8 hours': 8,
    '10-6 hours': 8,
    '55-66 hours': 8.5,  # probably per week
    '8-89 hours': 8.5,  # probably typo
    '8-9 hours': 8.5,
    '50-75 hours': 8.5,  # probably per week
    'More than 8 hours': 9,
    '60-65 hours': 9,  # probably per week
    '9-11 hours': 10,
    '10-11 hours': 10.5,
}

diet_dictionary = {
    'More Healty': 0,
    'Healthy': 1,
    '5 Healthy': 1,
    'Mealy': 2,
    'Less than Healthy': 2,
    'Less Healthy': 2,
    'Moderate': 3,
    '5 Unhealthy': 4,
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

cols_profession_to_delete = [
    'PhD', 'MBBS', 'B.Ed', 'M.Ed', 'BBA', 'MBA', 'LLM', 'BCA', 'B.Com', 'BE', 'Simran', '3M',
    'Name', 'No', '24th', 'Unveil', 'Unhealthy', 'Yuvraj', 'Yogesh', 'Patna', 'Nagpur',
    'Pranav', 'Visakhapatnam', 'Moderate', 'Manvi', 'Yogesh', 'Samar', 'Surat'
]


def handle_null(X):
    # delete weird professions
    X['Profession'] = X['Profession'].mask(X['Profession'].isin(cols_profession_to_delete))

    # create "student" profession (disabled, confuses the model)
    # X.loc[(X['Working_Professional_or_Student'] == 'Student') & (X['Profession'].isnull()), 'Profession'] = 'Student'

    # merge satisfaction and pressure columns and 0fill remainings (disabled, confuses the model)
    # X['Satisfaction'] = X['Study_Satisfaction'].fillna(X['Job_Satisfaction'])
    # X['Pressure'] = X['Academic_Pressure'].fillna(X['Work_Pressure'])
    # X['Satisfaction'] = X['Satisfaction'].fillna(0)
    # X['Pressure'] = X['Pressure'].fillna(0)
    # X.drop(['Study_Satisfaction'], axis=1, inplace=True)
    # X.drop(['Job_Satisfaction'], axis=1, inplace=True)
    # X.drop(['Academic_Pressure'], axis=1, inplace=True)
    # X.drop(['Work_Pressure'], axis=1, inplace=True)

    # fill metric numericals with values that make sense (eg: if working, then no study stress)
    X['Academic_Pressure'] = X['Academic_Pressure'].fillna(0)
    X['Work_Pressure'] = X['Work_Pressure'].fillna(0)
    X['Study_Satisfaction'] = X['Study_Satisfaction'].fillna(0)
    X['Job_Satisfaction'] = X['Job_Satisfaction'].fillna(0)
    X['CGPA'] = X['CGPA'].fillna(-1)  # we want to emphasize that it's a missing value
    X['Financial_Stress'] = X['Financial_Stress'].fillna(-1)  # we want to emphasize that it's a missing value

    # fill unknown categoricals
    X['Profession'] = X['Profession'].fillna('Unknown')
    X['Dietary_Habits'] = X['Dietary_Habits'].fillna('Unknown')
    X['Degree'] = X['Degree'].fillna('Unknown')

    # Assign values based on the dictionaries and set -1 for empty values
    X['Sleep_Duration'] = X['Sleep_Duration'].apply(lambda x: sleep_dictionary.get(x, -1) if x != '' else -1)
    X['Dietary_Habits'] = X['Dietary_Habits'].apply(lambda x: diet_dictionary.get(x, -1) if x != '' else -1)
    X['Gender'] = X['Gender'].apply(lambda x: gender_dictionary.get(x, -1) if x != '' else -1)
    X['Have_you_ever_had_suicidal_thoughts_'] = X['Have_you_ever_had_suicidal_thoughts_'].apply(
        lambda x: yes_no_dictionary.get(x, -1) if x != '' else -1)
    X['Working_Professional_or_Student'] = X['Working_Professional_or_Student'].apply(
        lambda x: student_profession_dictionary.get(x, -1) if x != '' else -1)
    X['Family_History_of_Mental_Illness'] = X['Family_History_of_Mental_Illness'].apply(
        lambda x: yes_no_dictionary.get(x, -1) if x != '' else -1)


def compute_hours_column(row, professional):
    if row['Working_Professional_or_Student'] == professional:
        return row['Work_Study_Hours']
    else:
        return 0


def compute_work_stress_column(row):
    if row['Working_Professional_or_Student'] == 1:
        return row['Financial_Stress'] + row['Work_Pressure'] - row['Job_Satisfaction']
    else:
        return 0


def compute_academic_stress_column(row):
    if row['Working_Professional_or_Student'] == 0:
        return row['Financial_Stress'] + row['Academic_Pressure'] - row['Study_Satisfaction']
    else:
        return 0


def create_features(X):
    # create additional stress columns
    X['Work_Stress'] = X.apply(lambda row: compute_work_stress_column(row), axis=1)
    X['Academic_Stress'] = X.apply(lambda row: compute_academic_stress_column(row), axis=1)

    # distinct study and work hours
    X['Work_Hours'] = X.apply(lambda row: compute_hours_column(row, 1), axis=1)
    X['Study_Hours'] = X.apply(lambda row: compute_hours_column(row, 0), axis=1)

    X.drop(['Work_Study_Hours'], axis=1, inplace=True)


def drop_features(X):
    X.drop(['Name'], axis=1, inplace=True)
    X.drop(['City'], axis=1, inplace=True)


def fix_name(degree):
    # if it's not a string, do nothing
    if not isinstance(degree, str):
        return degree

    # remove special characters
    degree = re.sub('[^A-Za-z0-9_]+', '', degree)

    # lowercase
    degree = degree.lower()

    return degree


def fix_categoricals(X):
    # standardize names to avoid duplicates and misspellings
    X['Degree'] = X['Degree'].apply(fix_name)
    X['Profession'] = X['Profession'].apply(fix_name)


class MentalHealthDataPreprocessor(DataPreprocessor):

    def preprocess_data(self, X: DataFrame):
        handle_null(X)
        # self.create_features(X)  # Disabled, does not improve CV
        drop_features(X)
        fix_categoricals(X)
