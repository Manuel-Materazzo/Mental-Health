import math
import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_inference_wrapper import ModelInferenceWrapper
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from abc import ABC, abstractmethod


def show_confusion_matrix(real_values: Series, predictions):
    cm = confusion_matrix(real_values, predictions)
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Real Data')
    plt.show()


def save_model(model: ModelInferenceWrapper):
    os.makedirs('../target', exist_ok=True)

    with open('../target/model.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model() -> ModelInferenceWrapper:
    with open('../target/model.pkl', 'rb') as file:
        return pickle.load(file)


class Trainer(ABC):
    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE):
        self.pipeline: DTPipeline = pipeline
        self.metric: AccuracyMetric = metric
        self.model_wrapper = model_wrapper
        self.evals: [] = []

    def get_pipeline(self) -> DTPipeline:
        """
        Returns the pipeline that gets used for training.
        :return:
        """
        return self.pipeline

    def get_model_name(self) -> str:
        return type(self.model_wrapper).__name__

    def show_feature_importance(self, X: DataFrame):
        # Apply the same trasformations as the training process
        processed_X = self.pipeline.transform(X)

        # Get columns and importance list
        features = list(processed_X.columns)
        importance_df = self.model_wrapper.get_feature_importance(features)

        print(importance_df)
        # plot it!
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feats')
        plt.show()

    def show_loss(self):
        if len(self.evals) == 0:
            print("ERROR: No model has been fitted with an evaluation set")
            return

        plt.figure(figsize=(12, 6))
        plt.xlabel('Iterations')
        plt.ylabel(self.metric.value)
        plt.title('Loss Over Iterations')

        i = 0

        # add a line to the plot for each training done
        for eval_round in self.evals:
            accuracy_metric_name = next(iter(eval_round['validation_0']))
            epochs = len(eval_round['validation_0'][accuracy_metric_name])
            x_axis = range(0, epochs)
            plt.plot(x_axis, eval_round['validation_0'][accuracy_metric_name], label='Split-{}'.format(i))
            i = i + 1

        plt.legend()
        plt.show()

    def train_model(self, train_X: DataFrame, train_y: Series, val_X: DataFrame = None, val_y: Series = None,
                    iterations=1000, params=None) -> ModelWrapper:
        """
        Trains a Wrapped Model on the provided training data.
        When validation data is provided, the model is trained with early stopping.
        :param params:
        :param train_X:
        :param train_y:
        :param val_X:
        :param val_y:
        :param iterations:
        :return:
        """
        params = params or {}
        processed_train_X = self.pipeline.fit_transform(train_X)

        # if we have validation sets, train with early stopping rounds
        if val_y is not None:
            processed_val_X = self.pipeline.transform(val_X)
            self.model_wrapper.train_until_optimal(processed_train_X, processed_val_X, train_y, val_y, params)
            self.evals.append(self.model_wrapper.get_loss())
        # else train with all the data
        else:
            self.model_wrapper.fit(processed_train_X, train_y, iterations, params)

        return self.model_wrapper

    @abstractmethod
    def validate_model(self, X: DataFrame, y: Series, log_level=1, iterations=None, params=None) -> (float, int):
        pass

    def get_predictions(self, X: DataFrame) -> Series:
        if self.metric == AccuracyMetric.AUC:
            return self.model_wrapper.predict_proba(X)
        else:
            return self.model_wrapper.predict(X)

    def calculate_accuracy(self, predictions: Series, real_values: Series) -> float:
        """
        Calculates the accuracy of the provided predictions, using the metric specified when creating the trainer.
        :param predictions:
        :param real_values:
        :return:
        """
        match self.metric:
            case AccuracyMetric.MAE:
                return mean_absolute_error(real_values, predictions)
            case AccuracyMetric.MSE:
                return mean_squared_error(real_values, predictions)
            case AccuracyMetric.RMSE:
                return math.sqrt(mean_squared_error(real_values, predictions))
            case AccuracyMetric.AUC:
                return roc_auc_score(real_values, predictions)
            case AccuracyMetric.Accuracy:
                return accuracy_score(real_values, predictions)
