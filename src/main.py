import pandas as pd
import re
import time

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.models.xgb_classifier import XGBClassifierWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.mental_health_dt_pipeline import MentalHealthDTPipeline
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.fast_cross_trainer import FastCrossTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.hyperparameter_optimizers.hyperopt_bayesian_optimizer import HyperoptBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.trainers.trainer import save_model


def load_data():
    # Load the data
    file_path = '../resources/train.csv'
    data = pd.read_csv(file_path)
    # standardize column names
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # Remove rows with missing target, separate target from predictors
    pruned_data = data.dropna(axis=0, subset=['Depression'])
    y = pruned_data['Depression']
    X = pruned_data.drop(['Depression'], axis=1)
    return X, y


print("Loading data...")
X, y = load_data()

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# instantiate data pipeline
pipeline = MentalHealthDTPipeline(X, True)

# pick a model, a trainer and an optimizer
model_type = XGBClassifierWrapper()
trainer = CachedAccurateCrossTrainer(pipeline, model_type, X, y, metric=AccuracyMetric.AUC)
optimizer = CustomGridOptimizer(trainer, model_type)

# optimize parameters
print("Tuning Hyperparameters...")
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03, direction=OptimizationDirection.MAXIMIZE)
print(optimized_params)
end = time.time()

print("Tuning took {} seconds".format(end - start))

print("Training and evaluating model...")
_, iterations = trainer.validate_model(X, y, log_level=1, params=optimized_params)
print()

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=iterations, params=optimized_params)

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
