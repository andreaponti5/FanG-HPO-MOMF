import numpy as np
import time
import torch

from botorch.test_functions.base import BaseTestProblem
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from problems.utils import statistical_parity_difference


class HpoMlp(BaseTestProblem):

    def __init__(self, dataset_full, dataset_half, target, sensitive_features, seed=None):
        self.target = target
        self.sensitive_features = sensitive_features
        self.features = list(set(dataset_full.columns) - {target})
        self.dataset = [dataset_half, dataset_full]
        self.X = [dataset_half[self.features], dataset_full[self.features]]
        self.y = [dataset_half[target], dataset_full[target]]
        self.seed = seed

        self.dim = 10 + 1
        self.num_objectives = 2

        self._bounds = [(-6, -1)]                                   # alpha
        self._bounds.extend([(-3, -0.001) for _ in range(2)])       # beta_1, beta_2
        self._bounds.extend([(2, 32) for _ in range(4)])            # layers neurons
        self._bounds.append((-6, -1))                               # learning_rate_init
        self._bounds.append((1, 4))                                 # n_layers
        self._bounds.append((-5, -2))                               # tol
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
        # TODO Verificare l'ordine e la scala degli iperparametri
        # Convert the array in the hyperparameter configuration
        n_layer = int(round(hyperparameter[8].tolist(), 0))
        hidden_layer_sizes = [int(round(log_value, 0))
                              for log_value in hyperparameter[3:n_layer + 3].tolist()]
        alpha = 10 ** hyperparameter[0].tolist()
        learning_rate_init = 10 ** hyperparameter[7].tolist()
        beta_1, beta_2 = 10 ** hyperparameter[1].tolist(), 10 ** hyperparameter[2].tolist()
        tol = 10 ** hyperparameter[9].tolist()
        source = int(round(hyperparameter[10].tolist(), 0))
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
            classifier = MLPClassifier(random_state=self.seed,
                                       hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                                       learning_rate_init=learning_rate_init, beta_1=beta_1,
                                       beta_2=beta_2, tol=tol).fit(X_train, y_train)
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
