# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import xDeepFM
from ..utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'dnn_hidden_units,cin_layer_size,cin_split_half,cin_activation,sparse_feature_num,dense_feature_dim',
    [((), (), True, 'linear', 1, 2),
     ((8,), (), True, 'linear', 1, 1),
     ((), (8,), True, 'linear', 2, 2),
     ((8,), (8,), False, 'relu', 2, 0)]
)
def test_xDeepFM(dnn_hidden_units, cin_layer_size, cin_split_half, cin_activation, sparse_feature_num,
                 dense_feature_dim):
    model_name = 'xDeepFM'

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=sparse_feature_num)
    model = xDeepFM(feature_columns, feature_columns, dnn_hidden_units=dnn_hidden_units, cin_layer_size=cin_layer_size,
                    cin_split_half=cin_split_half, cin_activation=cin_activation, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == '__main__':
    pass
