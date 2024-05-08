import unittest
import numpy as np
from pandas.core.frame import DataFrame
from pandas import Series
from scipy.sparse import spmatrix

import funcnodes as fn

# from typing import Tuple,
from funcnodes_sklearn.datasets import (
    _20newsgroups,
    _20newsgroups_vectorized,
    _20newsgroups_vectorized_as_frame,
    _california_housing,
    _california_housing_as_frame,
    _covtype,
    _covtype_as_frame,
    _kddcup99,
    _kddcup99_as_frame,
    _lfw_pairs,
    _lfw_people,
    _olivetti_faces,
    # _openml,
    _rcv1,
    # _species_distributions,
    _breast_cancer,
    _breast_cancer_as_frame,
    _diabetes,
    _diabetes_as_frame,
    _digits,
    # _digits_as_frame,
    # # _text_files,
    _iris,
    _iris_as_frame,
    _linnerud,
    _linnerud_as_frame,
    _sample_image,
    # # _svmlight_file
    _wine,
    _wine_as_frame,
    # _biclusters,
    # _blobs,
    # _checkerboard,
    # _circles,
    # _friedman1,
    # _friedman2,
    # _friedman3,
    # _gaussian_quantiles,
    # _hastie_10_2,
    # _low_rank_matrix,
    # _moons,
)


class Test20newsgroups(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _20newsgroups()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        self.assertIsInstance(data, list)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)


class Test20newsgroupsVectorized(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _20newsgroups_vectorized()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        self.assertIsInstance(data, spmatrix)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)


class Test20newsgroupsVectorizedAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _20newsgroups_vectorized_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)


class TestCaliforniaHousing(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _california_housing()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)


class TestCaliforniaHousingAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _california_housing_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)


class TestCovtype(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _covtype()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        feature_names = model.outputs["feature_names"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)
        self.assertIsInstance(feature_names, list)


class TestCovtypeAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _covtype_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        feature_names = model.outputs["feature_names"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)
        self.assertIsInstance(feature_names, list)


class TestKddcup99(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _kddcup99()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        feature_names = model.outputs["feature_names"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)
        self.assertIsInstance(feature_names, list)


class TestKddcup99AsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _kddcup99_as_frame()
        self.assertIsInstance(model, fn.Node)
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        target_names = model.outputs["target_names"].value
        feature_names = model.outputs["feature_names"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, list)
        self.assertIsInstance(feature_names, list)


class TestLfwPairs(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _lfw_pairs()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        pairs = model.outputs["pairs"].value
        target = model.outputs["target"].value
        target_names = model.outputs["target_names"].value
        DESCR = model.outputs["DESCR"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(pairs, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, np.ndarray)


class TestLfwPeople(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _lfw_people()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        images = model.outputs["images"].value
        target = model.outputs["target"].value
        target_names = model.outputs["target_names"].value
        DESCR = model.outputs["DESCR"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(target_names, np.ndarray)


class TestOlivettiFaces(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _olivetti_faces()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        images = model.outputs["images"].value
        target = model.outputs["target"].value
        DESCR = model.outputs["DESCR"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(DESCR, str)


# class TestOpenml(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _openml()
#         self.assertIsInstance(dataset, dict)
#         # self.assertEqual(
#         #     list(dataset.keys()),
#         #     ["data", "target", "frame", "target_names", "feature_names", "DESCR"],
#         # )
class TestRcv1(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _rcv1()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        sample_id = model.outputs["sample_id"].value
        target = model.outputs["target"].value
        target_names = model.outputs["target_names"].value
        DESCR = model.outputs["DESCR"].value
        self.assertIsInstance(data, spmatrix)
        self.assertIsInstance(target, spmatrix)
        self.assertIsInstance(sample_id, np.ndarray)
        self.assertIsInstance(target_names, np.ndarray)
        self.assertIsInstance(DESCR, str)


# class TestSpeciesDistributions(unittest.IsolatedAsyncioTestCase):
#     async def test_default_parameters(self):
#         model: fn.Node = _species_distributions()
#         self.assertIsInstance(model, fn.Node)
#         model.trigger()
#         await model
#         coverages = model.outputs["coverages"].value
#         train = model.outputs["train"].value
#         test = model.outputs["test"].value
#         Nx = model.outputs["Nx"].value
#         Ny = model.outputs["Ny"].value
#         x_left_lower_corner = model.outputs["x_left_lower_corner"].value
#         y_left_lower_corner = model.outputs["y_left_lower_corner"].value
#         grid_size = model.outputs["grid_size"].value
#         # self.assertIsInstance(coverages, np.ndarray) # TODO: check if this is a np.array
#         # self.assertIsInstance(train, np.ndarray) # TODO: check if this is a np.array
#         # self.assertIsInstance(test, np.ndarray) # TODO: check if this is a np.array
#         self.assertIsInstance(Nx, int)
#         self.assertIsInstance(Ny, int)
#         self.assertIsInstance(x_left_lower_corner, float)
#         self.assertIsInstance(y_left_lower_corner, float)
#         self.assertIsInstance(grid_size, float)


class TestBreastCancer(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _breast_cancer()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        feature_names = model.outputs["feature_names"].value
        target_names = model.outputs["target_names"].value
        DESCR = model.outputs["DESCR"].value
        filename = model.outputs["filename"].value

        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertIsInstance(feature_names, np.ndarray)
        self.assertIsInstance(target_names, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(filename, str)


class TestBreastCancerAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _breast_cancer_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        feature_names = model.outputs["feature_names"].value
        target_names = model.outputs["target_names"].value
        DESCR = model.outputs["DESCR"].value
        filename = model.outputs["filename"].value

        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)
        self.assertIsInstance(feature_names, np.ndarray)
        self.assertIsInstance(target_names, np.ndarray)
        self.assertIsInstance(DESCR, str)
        self.assertIsInstance(filename, str)


class TestDiabetes(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _diabetes()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value

        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)


class TestDiabetesAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _diabetes_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value

        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)


class TestDigits(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _digits()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)


class TestDigitsAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _diabetes_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)


# # class TestTextFiles(unittest.IsolatedAsyncioTestCase):
# #     def test_default_parameters(self):
# #         dataset = _text_files()
# #         self.assertIsInstance(dataset, dict)


class TestIris(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _iris()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)


class TestIrisAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _iris_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)


class TestLinnerud(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _linnerud()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)


class TestLinnerudAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _linnerud_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, DataFrame)


class TestSampleImage(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _sample_image()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        img = model.outputs["out"].value
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (427, 640, 3))


# # class TestSVMFile(unittest.IsolatedAsyncioTestCase):
# #     def test_default_parameters(self):
# #         dataset = _svmlight_file()
# #         self.assertIsInstance(dataset, Tuple)


class TestWine(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _wine()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(target, np.ndarray)


class TestWineAsFrame(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        model: fn.Node = _wine_as_frame()
        self.assertIsInstance(model, fn.Node)
        model.trigger()
        await model
        data = model.outputs["data"].value
        target = model.outputs["target"].value
        self.assertIsInstance(data, DataFrame)
        self.assertIsInstance(target, Series)


# class TestWine(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _wine()
#         self.assertIsInstance(dataset, dict)
#         self.assertEqual(list(dataset.target_names), ['class_0', 'class_1', 'class_2'])

# class TestBiClusters(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _biclusters(shape=(10, 10), n_clusters=5)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (10, 10))
#         self.assertEqual(dataset[1].shape, (5, 10))
#         self.assertEqual(dataset[2].shape, (5, 10))

# class TestBlobs(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _blobs()
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 2))
#         self.assertEqual(dataset[1].shape, (100,))

# class TestCheckerboard(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _checkerboard(shape=(10, 10), n_clusters=5)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (10, 10))
#         self.assertEqual(dataset[1].shape, (25, 10))
#         self.assertEqual(dataset[2].shape, (25, 10))
# class TestCircles(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _circles(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 2))
#         self.assertEqual(dataset[1].shape, (100,))

# class TestFriedman1(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _friedman1(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 10))
#         self.assertEqual(dataset[1].shape, (100,))

# class TestFriedman2(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _friedman2(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 4))
#         self.assertEqual(dataset[1].shape, (100,))

# class TestFriedman3(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _friedman3(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 4))
#         self.assertEqual(dataset[1].shape, (100,))


# class TestGaussianQuantiles(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _gaussian_quantiles(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 2))
#         self.assertEqual(dataset[1].shape, (100,))


# class TestHastie102(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _hastie_10_2(random_state=42)
#         self.assertIsInstance(dataset, Tuple)
#         self.assertEqual(dataset[0].shape, (100, 10))
#         self.assertEqual(dataset[1].shape, (100,))

# class TestLowRankMatrix(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _low_rank_matrix(random_state=42)
#         self.assertIsInstance(dataset, np.ndarray)
# #         self.assertEqual(dataset.shape, (100, 100))
# class TestMoons(unittest.IsolatedAsyncioTestCase):
#     def test_default_parameters(self):
#         dataset = _moons(random_state=42)
#         self.assertIsInstance(dataset, )
#         self.assertEqual(dataset.shape, (100, 100))
