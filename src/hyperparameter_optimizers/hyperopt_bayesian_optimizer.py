from pandas import DataFrame, Series
from hyperopt import STATUS_OK, Trials, fmin, STATUS_FAIL, tpe

from src.enums.optimization_direction import OptimizationDirection
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer


class HyperoptBayesianOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer, model_wrapper: ModelWrapper):
        super().__init__(trainer, model_wrapper)
        self.y = None
        self.X = None
        self.direction = OptimizationDirection.MINIMIZE
        self.domain_space = model_wrapper.get_bayesian_space()
        self.model_wrapper = model_wrapper

    def tune(self, X: DataFrame, y: Series, final_lr: float,
             direction: OptimizationDirection = OptimizationDirection.MINIMIZE) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a bayesian optimization
        Trains cross-validated model for each combination of hyperparameters, and picks the best based on MAE.
        :param direction:
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        self.X = X
        self.y = y
        self.direction = direction

        trials = Trials()

        best_hyperparams = fmin(fn=self.__objective,
                                space=self.domain_space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

        self.params.update(
            self.space_to_params(best_hyperparams)
        )

        self.params['learning_rate'] = final_lr

        return self.params

    def __objective(self, space):
        """
        Defines the objective function to be minimized.
        Trains a model with hyperparameters and returns the cross validated MAE.
        :return:
        """
        # hyperopt minimizes a loss function, in order to maximize we need to change the accuracy sign
        if self.direction == OptimizationDirection.MAXIMIZE:
            multiplier = -1
        elif self.direction == OptimizationDirection.MINIMIZE:
            multiplier = 1
        else:
            print("Optimization direction unknown, can't optimize")
            return {'status': STATUS_FAIL}

        params = self.space_to_params(space)
        accuracy, _ = self.trainer.validate_model(self.X, self.y, log_level=0, params=params)
        return {'loss': multiplier * accuracy, 'status': STATUS_OK}
