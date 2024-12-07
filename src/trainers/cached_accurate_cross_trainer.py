import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import trim_mean

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import Trainer


class CachedAccurateCrossTrainer(Trainer):
    """
    Wrapper of AccurateCrossTrainer that takes X and Y at initialization time and caches kfold splits.
    """

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, X: DataFrame, y: Series,
                 metric: AccuracyMetric = AccuracyMetric.MAE, grouping_columns: list[str] = None):
        super().__init__(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns)
        self.X = X
        self.y = y
        self.splits = self.__cache_splits()
        self.trainer = AccurateCrossTrainer(pipeline, model_wrapper)

    def __cache_splits(self) -> list:
        """
        Splits X and y into 5 folds at initialization time and returns them as a list.
        :return:
        """
        kf = self.get_kfold_type()

        splits = []

        if self.grouping_columns is not None:
            groups = self.X[self.grouping_columns]
        else:
            groups = None

        for train_index, val_index in kf.split(self.X, self.y, groups):
            # split train and validation data
            train_X, val_X = self.X.iloc[train_index], self.X.iloc[val_index]
            train_y, val_y = self.y.iloc[train_index], self.y.iloc[val_index]

            splits.append([train_X, val_X, train_y, val_y])

        return splits

    def __cross_train(self, split, iterations=None, params=None,
                      output_prediction_comparison=False) -> (int, int, DataFrame):

        # if no rounds, train with early stopping
        if iterations is None:
            self.trainer.train_model(split[0], split[2], split[1], split[3], params=params)
        # else train normally
        else:
            self.trainer.train_model(split[0], split[2], iterations=iterations, params=params)

        # re-process val_X to obtain accuracy
        processed_val_X = self.trainer.pipeline.transform(split[1])

        # Predict and calculate accuracy
        predictions = self.get_predictions(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, split[3])

        # create a dataframe with comparison
        if output_prediction_comparison:
            prediction_comparison = pd.DataFrame({'real_values': split[3], 'predictions': list(predictions)})
        else:
            prediction_comparison = None

        try:
            # number of boosting rounds used in the best model, accuracy
            return self.model_wrapper.get_best_iteration(), accuracy, prediction_comparison
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return iterations, accuracy, prediction_comparison

    def validate_model(self, X: DataFrame, y: Series, log_level=2, iterations=None, params=None,
                       output_prediction_comparison=False) -> (float, int, DataFrame):
        """
        Trains 5 Models on the provided training data by cross-validation.
        Data is splitted into 5 folds, each model is trained on 4 folds and validated on 1 fold.
        The validation fold is always different, so we are basically training and validating over the entire dataset.
        Accuracy score and optimal iterations of each model are then meaned to get overall values.
        If no rounds are provided, the models are trained using early stopping and will return the optimal number of
        boosting rounds alongside the Accuracy.

        :param output_prediction_comparison: whether to output a dataframe containing predictions and actual values.
        :param X:
        :param y:
        :param iterations:
        :param log_level:
        :param params:
        :return:
        """
        # Placeholder for cross-validation accuracy scores
        cv_scores = []
        best_rounds = []

        self.evals = []
        oof_comparisons_dataframes = []

        # Loop through each fold
        for split in self.splits:
            # cross train
            best_iteration, accuracy, oof_prediction_comparison = self.__cross_train(split, iterations=iterations,
                                                                                     params=params,
                                                                                     output_prediction_comparison=output_prediction_comparison)

            # when oof prediction save is enabled, add the prediction to the list
            if oof_prediction_comparison is not None:
                oof_comparisons_dataframes.append(oof_prediction_comparison)

            # add the best iteration and accuracy to lists
            best_rounds.append(best_iteration or 0)
            cv_scores.append(accuracy)

        # compute comparisons across all folds
        if len(oof_comparisons_dataframes) > 0:
            oof_prediction_comparisons = pd.concat(oof_comparisons_dataframes)
        else:
            oof_prediction_comparisons = None

        # extract evals
        self.evals = self.trainer.evals

        # Calculate the mean accuracy from cross-validation
        mean_accuracy = np.mean(cv_scores)
        # Calculate optimal boosting rounds
        optimal_boost_rounds = int(np.mean(best_rounds))
        pruned_optimal_boost_rounds = int(trim_mean(best_rounds, proportiontocut=0.1))  # trim extreme values

        if log_level > 0:
            print("Cross-Validation {}: {}".format(self.metric.value, mean_accuracy))
            if log_level > 1:
                print(cv_scores)
            print("Optimal iterations: ", optimal_boost_rounds)
            if log_level > 1:
                print("Pruned Optimal iterations: ", pruned_optimal_boost_rounds)
                print(best_rounds)

        # Cross validate model with the optimal boosting round, to check on accuracy discrepancies
        if iterations is None and log_level > 0:
            print("Generating {} with optimal iterations".format(self.metric.value))
            self.validate_model(X, y, iterations=optimal_boost_rounds, log_level=1, params=params)

        return mean_accuracy, optimal_boost_rounds, oof_prediction_comparisons
