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

from funcnodes_sklearn.fit import _fit, _fit_transform, _inverse_transform, _transform

class TestFittingingNodes(unittest.IsolatedAsyncioTestCase):
    async def test_fit_transform_one_hot_encoder(self):
        model: fn.Node = _one_hot_encoder()
        self.assertIsInstance(model, fn.Node)
        X =[['Male', 1], ['Female', 3], ['Female', 2]]
        
        # async def test_fit(self):
        ft_model: fn.Node = _fit()
        ft_model.inputs["model"].connect(model.outputs["out"])
        ft_model.inputs["X"].value = X
        self.assertIsInstance(ft_model, fn.Node)
        # await fn.run_until_complete(ft_model,model)
        # print(ft_model.outputs["out"])
        
        X_t = [['Female', 1], ['Male', 4]]
        t_model: fn.Node = _transform()
        t_model.inputs["model"].connect(ft_model.outputs["out"])
        t_model.inputs["X"].value = X_t
        self.assertIsInstance(t_model, fn.Node)
        
        await fn.run_until_complete(t_model,ft_model,model)
        out = t_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [[1., 0., 1., 0., 0.],[0., 1., 0., 0., 0.]])

    async def test_fit_inverse_transform_one_hot_encoder(self):
        model: fn.Node = _one_hot_encoder()
        self.assertIsInstance(model, fn.Node)
        X =[['Male', 1], ['Female', 3], ['Female', 2]]
        
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
        
        await fn.run_until_complete(it_model,ft_model,model)
        out = it_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [['Male', 1],[None, 2]])
        
        
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
        
        await fn.run_until_complete(t_model,ft_model,model)
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
        
        await fn.run_until_complete(it_model,ft_model,model)
        out = it_model.outputs["out"]
        self.assertEqual(out.value().tolist(), [1, 1, 2, 6])