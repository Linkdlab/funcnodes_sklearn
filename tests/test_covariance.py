from typing import Iterator, Tuple
import numpy as np
import unittest
from unittest.mock import patch, Mock, MagicMock
from sklearn.covariance import (
    EmpiricalCovariance,
    EllipticEnvelope,
    GraphicalLasso,
    GraphicalLassoCV,
)
from funcnodes_sklearn.covariance import (
    empirical_covariance,
    elliptical_envelpoe,
    graphical_lasso,
    Covariance,
    Mode,
    graphical_lasso_cv,
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from joblib import Memory


class TestEmpiricalCovariance(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
        self.real_cov = np.array([[0.8, 0.3], [0.3, 0.4]])
        self.rng = np.random.RandomState(0)
        self.rng.multivariate_normal(mean=[0, 0], cov=self.real_cov, size=500)

    def test_default_parameters(self):
        cov = empirical_covariance().fit(self.X)
        print(cov.covariance_.tolist())
        self.assertIsInstance(cov, EmpiricalCovariance)
        self.assertEqual(
            cov.covariance_.tolist(),
            [
                [1.2222222222222223, 2.8888888888888893],
                [2.8888888888888893, 7.555555555555555],
            ],
        )
        self.assertEqual(
            cov.location_.tolist(), [2.3333333333333335, 3.3333333333333335]
        )


class TestEllipticEnvelope(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
        self.real_cov = np.array([[0.8, 0.3], [0.3, 0.4]])
        self.rng = np.random.RandomState(0)
        self.rng.multivariate_normal(mean=[0, 0], cov=self.real_cov, size=500)

    def test_default_parameters(self):
        cov = elliptical_envelpoe(random_state=0).fit(self.X)
        self.assertIsInstance(cov, EllipticEnvelope)
        self.assertEqual(
            cov.covariance_.tolist(),
            [
                [1.2222222222222223, 2.8888888888888893],
                [2.8888888888888893, 7.555555555555555],
            ],
        )
        self.assertEqual(
            cov.location_.tolist(), [2.3333333333333335, 3.3333333333333335]
        )

    def test_predict(self):
        cov = elliptical_envelpoe(random_state=0).fit(self.X)
        self.assertIsInstance(cov, EmpiricalCovariance)
        self.assertEqual(cov.predict([[0, 0], [3, 3]]).tolist(), [-1, -1])


class TestGraphicalLasso(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
        self.real_cov = np.array(
            [
                [0.8, 0.0, 0.2, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.2, 0.0, 0.3, 0.1],
                [0.0, 0.0, 0.1, 0.7],
            ]
        )
        self.rng = np.random.RandomState(0)
        self.rng.multivariate_normal(mean=[0, 0, 0, 0], cov=self.real_cov, size=200)

    def test_default_parameters(self):
        cov = graphical_lasso().fit(self.X)
        self.assertIsInstance(cov, GraphicalLasso)
        self.assertEqual(
            np.around(cov.covariance_, decimals=2).tolist(),
            [[1.22, 2.88], [2.88, 7.56]],
        )
        self.assertEqual(np.around(cov.location_, decimals=2).tolist(), [2.33, 3.33])

    def test_mode(self):
        cov = graphical_lasso(mode=Mode.LARS.value).fit(self.X)
        self.assertIsInstance(cov, GraphicalLasso)
        self.assertEqual(
            np.around(cov.covariance_, decimals=2).tolist(),
            [[1.22, 2.88], [2.88, 7.56]],
        )
        self.assertEqual(np.around(cov.location_, decimals=2).tolist(), [2.33, 3.33])

    def test_covariance(self):
        cov = graphical_lasso(covariance=Covariance.PRECOMPUTED.value).fit(
            self.real_cov
        )
        self.assertIsInstance(cov, GraphicalLasso)
        print(cov.covariance_.tolist())
        self.assertEqual(
            np.around(cov.covariance_, decimals=2).tolist(),
            [
                [0.8, 0.0, 0.19, 0.01],
                [0.0, 0.4, 0.0, 0.0],
                [0.19, 0.0, 0.3, 0.09],
                [0.01, 0.0, 0.09, 0.7],
            ],
        )


class TestGraphicalLassoCV(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
        self.real_cov = np.array(
            [
                [0.8, 0.0, 0.2, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.2, 0.0, 0.3, 0.1],
                [0.0, 0.0, 0.1, 0.7],
            ]
        )
        self.rng = np.random.RandomState(0)
        self.rng.multivariate_normal(mean=[0, 0, 0, 0], cov=self.real_cov, size=200)

    def test_default_parameters(self):
        cov = graphical_lasso_cv().fit(self.X)
        self.assertIsInstance(cov, GraphicalLassoCV)
        self.assertEqual(
            np.around(cov.covariance_, decimals=2).tolist(),
            [[1.22, 2.84], [2.84, 7.56]],
        )
        self.assertEqual(np.around(cov.location_, decimals=2).tolist(), [2.33, 3.33])

    def test_mode(self):
        cov = graphical_lasso_cv(mode=Mode.LARS.value).fit(self.X)
        self.assertIsInstance(cov, GraphicalLassoCV)
        self.assertEqual(
            np.around(cov.covariance_, decimals=2).tolist(),
            [[1.22, 2.84], [2.84, 7.56]],
        )
        self.assertEqual(np.around(cov.location_, decimals=2).tolist(), [2.33, 3.33])

    def test_cv(self):
        def generate_random_splits(
            num_splits: int, dataset_size: int, train_size: float = 0.8
        ) -> Iterator[Tuple[np.ndarray[int], np.ndarray[int]]]:
            for _ in range(num_splits):
                indices = np.random.permutation(dataset_size)
                train_indices = indices[: int(train_size * dataset_size)]
                test_indices = indices[int(train_size * dataset_size) :]
                yield train_indices, test_indices

        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42
        )
        num_splits = 2
        dataset_size = len(self.X)
        splits = generate_random_splits(num_splits, dataset_size)
        cov = graphical_lasso_cv(cv=splits).fit(X_train, y_train)
        self.assertIsInstance(cov, GraphicalLassoCV)
        # print(cov.covariance_.tolist())

        # self.assertEqual(
        #     cov.covariance_.tolist(),
        #     [
        #         [1.5742305894353745, 0.2992187179323442],
        #         [0.2992187179323442, 1.2481585144371685],
        #     ],
        # )
