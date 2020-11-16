# -*- coding: utf-8 -*-
from deepctr_torch.layers import activation
from tests.utils import layer_test


def test_dice():
    layer_test(activation.Dice, kwargs={'emb_size': 3, 'dim': 2},
               input_shape=(5, 3), expected_output_shape=(5,3))
    layer_test(activation.Dice, kwargs={'emb_size': 10, 'dim': 3},
               input_shape=(5, 3, 10), expected_output_shape=(5,3,10))

