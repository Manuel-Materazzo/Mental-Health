import pandas as pd
import re
import time

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.models.xgb_classifier import XGBClassifierWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.mental_health_dt_pipeline import MentalHealthDTPipeline
from src.preprocessors.mental_health_data_preprocessor import MentalHealthDataPreprocessor
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.trainers.trainer import save_model


def load_data():
    # Load the data
    file_path = '../resources/train.csv'
    data = pd.read_csv(file_path)

    # Remove rows with missing target, separate target from predictors
    data.dropna(axis=0, subset=['Depression'], inplace=True)
    y = data['Depression']
    X = data.drop(['Depression'], axis=1)

    # standardize column names
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    return X, y


print("Loading data...")
X, y = load_data()

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# instantiate preprocessor
preprocessor = MentalHealthDataPreprocessor()

# preprocess data
preprocessor.preprocess_data(X)

# instantiate pipeline
pipeline = MentalHealthDTPipeline(X)

# pick a model, a trainer and an optimizer
model_type = XGBClassifierWrapper(early_stopping_rounds=50)
trainer = CachedAccurateCrossTrainer(pipeline, model_type, X, y, metric=AccuracyMetric.AUC)
optimizer = CustomGridOptimizer(trainer, model_type, direction=OptimizationDirection.MAXIMIZE)

# optimize parameters
print("Tuning Hyperparameters...")
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
print(optimized_params)
end = time.time()

print("Tuning took {} seconds".format(end - start))

print("Training and evaluating model...")
_, iterations, _ = trainer.validate_model(X, y, log_level=1, params=optimized_params)
print()

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=iterations, params=optimized_params)

# save preprocessor on target directory
print("Saving preprocessor...")
preprocessor.save_preprocessor()

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
