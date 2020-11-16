# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, SparseFeat, DenseFeat, VarLenSparseFeat
from ..layers import SENETLayer, BilinearInteraction, DNN


class FiBiNET(BaseModel):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bilinear_type: str,bilinear function type used in Bilinear Interaction Layer,can be ``'all'`` , ``'each'`` or ``'interaction'``
    :param reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, bilinear_type='interaction',
                 reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu'):
        super(FiBiNET, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device)
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.filed_size = len(self.embedding_dict)
        self.SE = SENETLayer(self.filed_size, reduction_ratio, seed, device)
        self.Bilinear = BilinearInteraction(self.filed_size, self.embedding_size, bilinear_type, seed, device)
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        field_size = len(sparse_feature_columns)

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        embedding_size = sparse_feature_columns[0].embedding_dim
        sparse_input_dim = field_size * (field_size - 1) * embedding_size
        input_dim = 0

        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim

        return input_dim

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)

        senet_output = self.SE(sparse_embedding_input)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input)

        linear_logit = self.linear_model(X)
        temp = torch.split(torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1)
        dnn_input = combined_dnn_input(temp, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        if len(self.linear_feature_columns) > 0 and len(self.dnn_feature_columns) > 0:  # linear + dnn
            final_logit = linear_logit + dnn_logit
        elif len(self.linear_feature_columns) == 0:
            final_logit = dnn_logit
        elif len(self.dnn_feature_columns) == 0:
            final_logit = linear_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred
