import unittest
import numpy as np
import funcnodes as fn
from sklearn.base import BaseEstimator
from funcnodes_sklearn.preprocessing import (
    _binarizer,
    _function_transformer,
    _kbins_discretizer,
    _kbins_centerer,
    _label_binarizer,
    _label_encoder,
    _max_abs_scaler,
    _min_max_scaler,
    _normalizer,
    _one_hot_encoder,
)
from funcnodes_sklearn.decomposition import _pca
from funcnodes_sklearn.fit import (
    _fit,
    _fit_transform,
    _inverse_transform,
    _transform,
    _predict,
)
from funcnodes_sklearn.discriminant_analysis import _lda, _qda

_features_train = r"C:\Users\mo35pid\Documents\Work_Jena\dev\DataAnalysis\Image\pipline_ml\Systematic_Study_data_BA\Train_Test_BA_Gaussian0.5\Train_no_norm_Maxpool.npy"
_features_test = r"C:\Users\mo35pid\Documents\Work_Jena\dev\DataAnalysis\Image\pipline_ml\Systematic_Study_data_BA\Train_Test_BA_Gaussian0.5\Test_no_norm_Maxpool.npy"
_train_lbls = r"C:\Users\mo35pid\Documents\Work_Jena\dev\DataAnalysis\Image\pipline_ml\Systematic_Study_data_BA\labels\train_lbls.npy"
_test_lbls = r"C:\Users\mo35pid\Documents\Work_Jena\dev\DataAnalysis\Image\pipline_ml\Systematic_Study_data_BA\labels\test_lbls.npy"


class TestFittingingNodes(unittest.IsolatedAsyncioTestCase):
    async def test_fit_transform_one_hot_encoder(self):
        model: fn.Node = _one_hot_encoder()
        self.assertIsInstance(model, fn.Node)
        X = [["Male", 1], ["Female", 3], ["Female", 2]]

        # async def test_fit(self):
        ft_model: fn.Node = _fit()
        ft_model.inputs["model"].connect(model.outputs["out"])
        ft_model.inputs["X"].value = X
        self.assertIsInstance(ft_model, fn.Node)
        # await fn.run_until_complete(ft_model,model)
        # print(ft_model.outputs["out"])

        X_t = [["Female", 1], ["Male", 4]]
        t_model: fn.Node = _transform()
        t_model.inputs["model"].connect(ft_model.outputs["out"])
        t_model.inputs["X"].value = X_t
        self.assertIsInstance(t_model, fn.Node)

        await fn.run_until_complete(t_model, ft_model, model)
        out = t_model.outputs["out"]
        self.assertEqual(
            out.value().tolist(), [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        )

    async def test_fit_inverse_transform_one_hot_encoder(self):
        model: fn.Node = _one_hot_encoder()
        self.assertIsInstance(model, fn.Node)
        X = [["Male", 1], ["Female", 3], ["Female", 2]]

        # async def test_fit(self):
        ft_model: fn.Node = _fit()
        ft_model.inputs["model"].connect(model.outputs["out"])
        ft_model.inputs["X"].value = X
        self.assertIsInstance(ft_model, fn.Node)

        X_it = [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]
        it_model = _inverse_transform()
        it_model.inputs["model"].connect(ft_model.outputs["out"])
        it_model.inputs["X"].value = X_it
        self.assertIsInstance(it_model, fn.Node)

        await fn.run_until_complete(it_model, ft_model, model)
        out = it_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [["Male", 1], [None, 2]])

    async def test_fit_transform_label_encoder(self):
        model: fn.Node = _label_encoder()
        self.assertIsInstance(model, fn.Node)
        X = [1, 2, 2, 6]

        # async def test_fit(self):
        ft_model: fn.Node = _fit()
        ft_model.inputs["model"].connect(model.outputs["out"])
        ft_model.inputs["X"].value = X
        self.assertIsInstance(ft_model, fn.Node)
        # await fn.run_until_complete(ft_model,model)
        # print(ft_model.outputs["out"])

        X_t = [1, 1, 2, 6]
        t_model: fn.Node = _transform()
        t_model.inputs["model"].connect(ft_model.outputs["out"])
        t_model.inputs["X"].value = X_t
        self.assertIsInstance(t_model, fn.Node)

        await fn.run_until_complete(t_model, ft_model, model)
        out = t_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [0, 0, 1, 2])

    async def test_fit_inverse_transform_label_encoder(self):
        model: fn.Node = _label_encoder()
        self.assertIsInstance(model, fn.Node)
        X = [1, 2, 2, 6]

        # async def test_fit(self):
        ft_model: fn.Node = _fit()
        ft_model.inputs["model"].connect(model.outputs["out"])
        ft_model.inputs["X"].value = X
        self.assertIsInstance(ft_model, fn.Node)

        X_it = [0, 0, 1, 2]
        it_model = _inverse_transform()
        it_model.inputs["model"].connect(ft_model.outputs["out"])
        it_model.inputs["X"].value = X_it
        self.assertIsInstance(it_model, fn.Node)

        await fn.run_until_complete(it_model, ft_model, model)
        out = it_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [1, 1, 2, 6])

    async def test_pca_lda(self):
        x = np.load(_features_train)
        y = np.load(_features_test)
        x_labels = np.load(_train_lbls)
        y_labels = np.load(_test_lbls)

        le: fn.Node = _label_encoder()
        self.assertIsInstance(le, fn.Node)

        le_fit: fn.Node = _fit()
        le_fit.inputs["model"].connect(le.outputs["out"])
        le_fit.inputs["X"].value = x_labels
        self.assertIsInstance(le_fit, fn.Node)

        pca: fn.Node = _pca()
        pca.inputs["n_components"].value = "40"
        self.assertIsInstance(pca, fn.Node)

        pca_fit: fn.Node = _fit()
        pca_fit.inputs["model"].connect(pca.outputs["out"])
        pca_fit.inputs["X"].value = x
        self.assertIsInstance(pca_fit, fn.Node)
        # await fn.run_until_complete(pca_fit,model)
        # print(pca_fit.outputs["out"])

        # X_t = [1, 1, 2, 6]
        pca_y_transform: fn.Node = _transform()
        pca_y_transform.inputs["model"].connect(pca_fit.outputs["out"])
        pca_y_transform.inputs["X"].value = y
        self.assertIsInstance(pca_y_transform, fn.Node)

        pca_x_transform: fn.Node = _transform()
        pca_x_transform.inputs["model"].connect(pca_fit.outputs["out"])
        pca_x_transform.inputs["X"].value = x
        self.assertIsInstance(pca_x_transform, fn.Node)

        le_y_transform: fn.Node = _transform()
        le_y_transform.inputs["model"].connect(le_fit.outputs["out"])
        le_y_transform.inputs["X"].value = y_labels
        self.assertIsInstance(le_y_transform, fn.Node)

        le_x_transform: fn.Node = _transform()
        le_x_transform.inputs["model"].connect(le_fit.outputs["out"])
        le_x_transform.inputs["X"].value = x_labels
        self.assertIsInstance(le_x_transform, fn.Node)

        lda: fn.Node = _lda()
        self.assertIsInstance(lda, fn.Node)

        lda_fit: fn.Node = _fit()
        lda_fit.inputs["model"].connect(lda.outputs["out"])
        lda_fit.inputs["X"].connect(pca_x_transform.outputs["out"])
        lda_fit.inputs["y"].value = x_labels
        self.assertIsInstance(lda_fit, fn.Node)

        # lda_predict: fn.Node = _predict()
        # lda_predict.inputs["model"].connect(lda.outputs["out"])
        # lda_predict.inputs["X"].connect(pca_x_transform.outputs["out"])
        # self.assertIsInstance(lda_predict, fn.Node)

        await fn.run_until_complete(
            le_x_transform,
            le_y_transform,
            pca_x_transform,
            pca_y_transform,
            pca_fit,
            pca,
            le,
            lda,
            lda_fit,
            # lda_predict,
        )
