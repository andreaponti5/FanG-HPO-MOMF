import numpy as np
import time
import torch
import xgboost as xgb

from botorch.test_functions.base import BaseTestProblem

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from problems.utils import statistical_parity_difference


class HpoXgb(BaseTestProblem):

    def __init__(self, dataset_full, dataset_half, target, sensitive_features, seed=None):
        self.target = target
        self.sensitive_features = sensitive_features
        self.features = list(set(dataset_full.columns) - {target})
        self.dataset = [dataset_half, dataset_full]
        self.X = [dataset_half[self.features], dataset_full[self.features]]
        self.y = [dataset_half[target], dataset_full[target]]
        self.seed = seed

        self.dim = 7 + 1
        self.num_objectives = 2

        self._bounds = [(1, 256)]                                   # n_estimators
        self._bounds.append((-2, 0))                                # learning_rate
        self._bounds.append((0, 0.1))                               # gamma
        self._bounds.append((-3, 3))                                # reg_alpha
        self._bounds.append((-3, 3))                                # reg_lambda
        self._bounds.append((0.01, 1))                              # subsample
        self._bounds.append((1, 16))                                # max_depth
        self._bounds.append((0, 1))                                 # source

        self.ref_point = torch.Tensor([0, 0])

        super().__init__()

    def evaluate_true(self, X):
        f = []
        for hyperparameter in X:
            res = self.run_kfold(hyperparameter)
            f1 = np.mean(res['accuracy'])
            f2 = 1 - np.mean(res['dsp'])
            f.append([f1, f2])
        return torch.Tensor(f)

    def run_kfold(self, hyperparameter):
        # Convert the array in the hyperparameter configuration
        n_estimators = int(round(hyperparameter[0].tolist(), 0))
        learning_rate = 10 ** hyperparameter[1].tolist()
        gamma = hyperparameter[2].tolist()
        reg_alpha = 10 ** hyperparameter[3].tolist()
        reg_lambda = 10 ** hyperparameter[4].tolist()
        subsample = hyperparameter[5].tolist()
        max_depth = int(round(hyperparameter[6].tolist(), 0))
        source = int(round(hyperparameter[7].tolist(), 0))
        # Performance Metrics
        res = {'accuracy': [], 'dsp': [], 'train_time': []}
        # Run kfold
        kf = StratifiedKFold(n_splits=10)
        for train_idx, test_idx in kf.split(self.X[source], self.y[source]):
            # Divide train/test
            X_train, y_train = self.X[source].loc[train_idx], self.y[source].loc[train_idx]
            X_test, y_test = self.X[source].loc[test_idx], self.y[source].loc[test_idx]
            start = time.time()
            # Train the classifier and predict on test set
            classifier = xgb.XGBClassifier(n_jobs=-1,
                                           objective="binary:logistic", random_state=self.seed,
                                           n_estimators=n_estimators, learning_rate=learning_rate,
                                           gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                           subsample=subsample, max_depth=max_depth,
                                           verbosity=0).fit(X_train, y_train)
            res['train_time'].append(time.time() - start)
            y_pred = classifier.predict(X_test)
            # Compute accuracy and DSP
            res['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_dsp = []
            for feature in self.sensitive_features:
                f = X_test[feature].to_numpy()
                fold_dsp.append(statistical_parity_difference(y_pred, f))
            res['dsp'].append(fold_dsp)
        return res
