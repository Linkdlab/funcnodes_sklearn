import unittest

# from funcnodes import Shelf, NodeDecorator
from sklearn.datasets import make_classification
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from funcnodes_sklearn.calibration import (
    calibrated_classifier_cv,
    calibrationcurve,
    Method,
    Strategy,
)

from typing import Iterator, Tuple
import numpy as np


def generate_random_splits(
    num_splits: int, dataset_size: int, train_size: float = 0.8
) -> Iterator[Tuple[np.ndarray[int], np.ndarray[int]]]:
    for _ in range(num_splits):
        indices = np.random.permutation(dataset_size)
        train_indices = indices[: int(train_size * dataset_size)]
        test_indices = indices[int(train_size * dataset_size) :]
        yield train_indices, test_indices


class TestCalibratedClassifierCV(unittest.TestCase):
    def test_default_parameters(self):
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        calibrated_clf = calibrated_classifier_cv()
        self.assertIsInstance(calibrated_clf, ClassifierMixin)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        calibrated_clf.fit(X_train, y_train)
        y_pred = calibrated_clf.predict(X_test)
        self.assertEqual(len(calibrated_clf.calibrated_classifiers_), 5)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.5)

    def test_integer_cv(self):
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        _cv = 10
        calibrated_clf = calibrated_classifier_cv(cv=_cv)
        self.assertIsInstance(calibrated_clf, ClassifierMixin)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        calibrated_clf.fit(X_train, y_train)
        y_pred = calibrated_clf.predict(X_test)
        self.assertEqual(len(calibrated_clf.calibrated_classifiers_), _cv)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.5)

    def test_iterable_cv(self):
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42
        )
        ## Example usage with StratifiedKFold
        # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # splits = kf

        ## Example usage with KFold
        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # splits = kf.split(X_train)

        # Example usage with custom splits
        num_splits = 5
        dataset_size = len(X_train)
        splits = generate_random_splits(num_splits, dataset_size)

        calibrated_clf = calibrated_classifier_cv(cv=splits)
        self.assertIsInstance(calibrated_clf, ClassifierMixin)

        calibrated_clf.fit(X_train, y_train)
        y_pred = calibrated_clf.predict(X_test)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.5)

    def test_string_cv(self):
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        _cv = "prefit"

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        calibrated_clf = calibrated_classifier_cv(
            estimator=GaussianNB().fit(X_train, y_train), cv=_cv
        )  # TODO: Should it raise an error if _cv is this
        calibrated_clf.fit(X_train, y_train)
        y_pred = calibrated_clf.predict(X_test)
        self.assertIsInstance(calibrated_clf, ClassifierMixin)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.5)

    def test_isotonic_calibration(self):
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        self.assertIsInstance(Method.ISOTONIC.value, str)
        calibrated_clf = calibrated_classifier_cv(method=Method.ISOTONIC.value)
        self.assertIsInstance(calibrated_clf, ClassifierMixin)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        calibrated_clf.fit(X_train, y_train)
        y_pred = calibrated_clf.predict(X_test)
        self.assertEqual(len(calibrated_clf.calibrated_classifiers_), 5)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.5)


class TestCalibrationCurve(unittest.TestCase):
    def test_default_parameters(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
        prob_true, prob_pred = calibrationcurve(y_true, y_prob)
        self.assertIsInstance(prob_true, np.ndarray)
        self.assertIsInstance(prob_pred, np.ndarray)

    def test_strategy(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
        prob_true, prob_pred = calibrationcurve(
            y_true, y_prob, n_bins=3, strategy=Strategy.QUANTILE.value
        )
        self.assertIsInstance(prob_true, np.ndarray)
        self.assertIsInstance(prob_pred, np.ndarray)

    def test_poslabel(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
        prob_true, prob_pred = calibrationcurve(y_true, y_prob, n_bins=3, pos_label=1.1)
        self.assertIsInstance(prob_true, np.ndarray)
        self.assertIsInstance(prob_pred, np.ndarray)
