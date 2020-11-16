# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import DCN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_device


@pytest.mark.parametrize(
    'embedding_size,cross_num,hidden_size,sparse_feature_num,cross_parameterization',
    [(8, 0, (32,), 2, 'vector'), (8, 0, (32,), 2, 'matrix'),
     ]  # ('auto', 1, (32,), 3) , ('auto', 1, (), 1), ('auto', 1, (32,), 3)
)
def test_DCN(embedding_size, cross_num, hidden_size, sparse_feature_num, cross_parameterization):
    model_name = "DCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=sparse_feature_num)

    model = DCN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, cross_num=cross_num,
                cross_parameterization=cross_parameterization,
                dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
