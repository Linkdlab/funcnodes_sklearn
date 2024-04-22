from typing import Iterator, Tuple
import numpy as np
import unittest

# from unittest.mock import patch, Mock, MagicMock
from sklearn.base import RegressorMixin, TransformerMixin

# from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from funcnodes_sklearn.cross_decomposition import (
    cca,
    pls_canonical,
    Algorithm,
    pls_regression,
    pls_svd,
)


class TestCCA(unittest.TestCase):
    def setUp(self):
        self.X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]]
        self.Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    def test_default_parameters(self):
        cross_decomposition = cca().fit(self.X, self.Y)
        transform = cross_decomposition.transform(self.X, self.Y)
        self.assertIsInstance(cross_decomposition, RegressorMixin)
        self.assertIsInstance(transform, Tuple)


class TestPLSCanonical(unittest.TestCase):
    def setUp(self):
        self.X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]]
        self.Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    def test_default_parameters(self):
        cross_decomposition = pls_canonical().fit(self.X, self.Y)
        self.assertIsInstance(cross_decomposition, RegressorMixin)

    def test_algorithm(self):
        cross_decomposition = pls_canonical(algorithm=Algorithm.SVD.value).fit(
            self.X, self.Y
        )
        self.assertIsInstance(cross_decomposition, RegressorMixin)


class TestPLSRegression(unittest.TestCase):
    def setUp(self):
        self.X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]]
        self.Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    def test_default_parameters(self):
        cross_decomposition = pls_regression().fit(self.X, self.Y)
        transform = cross_decomposition.transform(self.X, self.Y)
        self.assertIsInstance(cross_decomposition, RegressorMixin)
        self.assertIsInstance(transform, Tuple)


class TestPLSSVD(unittest.TestCase):
    def setUp(self):
        self.X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]]
        self.Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    def test_default_parameters(self):
        cross_decomposition = pls_svd().fit(self.X, self.Y)
        X_c, Y_c = cross_decomposition.transform(self.X, self.Y)
        self.assertIsInstance(cross_decomposition, TransformerMixin)
        self.assertEqual(X_c.shape, (4, 2))
        self.assertEqual(Y_c.shape, (4, 2))
