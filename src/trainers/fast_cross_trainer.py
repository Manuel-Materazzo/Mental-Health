import xgboost as xgb
from pandas import DataFrame, Series

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.models.xgb_regressor import XGBRegressorWrapper
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class FastCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE):
        super().__init__(pipeline, model_wrapper, metric=metric)

    def validate_model(self, X: DataFrame, y: Series, iterations=1000, log_level=1, params=None) -> (float, int):
        """
        Trains 5 XGBoost regressors on the provided training data by cross-validation.
        This method uses default xgb.cv strategy for cross-validation.
        X is preprocessed using fit_transform on the pipeline, this will probably cause
        "Train-Test Contamination Data Leakage" and provide a MAE estimate with lower accuracy.
        :param X:
        :param y:
        :param iterations:
        :param params:
        :return:
        """
        if not isinstance(self.model_wrapper, XGBRegressorWrapper):
            print("ERROR: This trainer only works with XGBoost regressors")
            return 0, 0

        print("WARNING: using simple_cross_validation can cause train data leakage, prefer cross_validation or "
              "classic_validation instead")

        processed_X = self.pipeline.fit_transform(X)

        dtrain = xgb.DMatrix(processed_X, label=y)

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=iterations,
            nfold=5,
            metrics=self.metric.value.lower(),
            early_stopping_rounds=5,
            seed=0,
            as_pandas=True)

        self.evals = []

        # for each split
        for i in range(5):
            # extract Series of accuracy for the split
            split_accuracy = cv_results[f'test-{self.metric.value.lower()}-mean'] + cv_results[
                f'test-{self.metric.value.lower()}-std'] * i
            # format dictionary to be standard compliant
            self.evals.append({
                'validation_0': {
                    'rmse': split_accuracy
                }
            })

        # Extract the mean of the accuracy from cross-validation results
        # optimal point (iteration) where the model achieved its best performance
        accuracy = cv_results['test-' + self.metric.value.lower() + '-mean'].min()
        # if you train the model again, same seed, no early stopping, you can put this index as num_boost_round to get same result
        best_round = cv_results['test-' + self.metric.value.lower() + '-mean'].idxmin()

        if log_level > 0:
            print("#{} Cross-Validation {}: {}".format(best_round, self.metric.value, accuracy))

        return accuracy, best_round
