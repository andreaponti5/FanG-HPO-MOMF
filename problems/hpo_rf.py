import numpy as np
import time
import torch

from botorch.test_functions.base import BaseTestProblem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import Tensor

from problems.utils import statistical_parity_difference


class HpoRf(BaseTestProblem):

    def __init__(self, dataset_full, dataset_redux, target, sensitive_features, seed=None):
        self.target = target
        self.sensitive_features = sensitive_features
        self.features = list(set(dataset_full.columns) - {target})
        self.dataset = [dataset_redux, dataset_full]
        self.X = [dataset_redux[self.features], dataset_full[self.features]]
        self.y = [dataset_redux[target], dataset_full[target]]
        self.seed = seed

        self.dim = 2 + 1
        self.num_objectives = 2

        self._bounds = [(2, len(self.features))]    # max_features
        self._bounds.append((100, 1000))            # n_estimators
        self._bounds.append((0, 1))                 # fidelity

        self.ref_point = torch.Tensor([0, 0])

        self.train_time = []

        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = []
        for hyperparameter in X:
            res = self.run_kfold(hyperparameter)
            f1 = np.mean(res['accuracy'])
            f2 = 1 - np.max(np.mean(res["dsp"], axis=0))
            self.train_time.append(np.sum(res["train_time"]))
            # print(f"accuracy: {f1}, dsp: {f2}")
            f.append([f1, f2])
        return torch.Tensor(f)

    def run_kfold(self, hyperparameter):
        # TODO Verificare l'ordine e la scala degli iperparametri
        # Convert the array in the hyperparameter configuration
        max_features = int(round(hyperparameter[0].tolist(), 0))
        n_estimators = int(round(hyperparameter[1].tolist(), 0))
        fidelity = int(round(hyperparameter[2].tolist(), 0))
        # Performance Metrics
        res = {'accuracy': [], 'dsp': [], 'train_time': []}
        # Run kfold
        kf = StratifiedKFold(n_splits=10)
        for train_idx, test_idx in kf.split(self.X[fidelity], self.y[fidelity]):
            # Divide train/test
            X_train, y_train = self.X[fidelity].loc[train_idx], self.y[fidelity].loc[train_idx]
            X_test, y_test = self.X[fidelity].loc[test_idx], self.y[fidelity].loc[test_idx]
            start = time.time()
            # Train the classifier and predict on test set
            classifier = RandomForestClassifier(n_jobs=-1, random_state=self.seed, n_estimators=n_estimators,
                                                max_features=max_features).fit(X_train, y_train)
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
