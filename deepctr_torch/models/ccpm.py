# -*- coding:utf-8 -*-
"""

Author:
    Zeng Kai,kk163mail@126.com

Reference:
    [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
    (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)

"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..layers.core import DNN
from ..layers.interaction import ConvLayer
from ..layers.utils import concat_fun


class CCPM(BaseModel):
    """Instantiates the Convolutional Click Prediction Model architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, conv_kernel_width=(6, 5),
                 conv_filters=(4, 4),
                 dnn_hidden_units=(256,), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', dnn_use_bn=False, dnn_activation='relu'):

        super(CCPM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device)

        if len(conv_kernel_width) != len(conv_filters):
            raise ValueError(
                "conv_kernel_width must have same element with conv_filters")

        filed_size = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        self.conv_layer = ConvLayer(field_size=filed_size, conv_kernel_width=conv_kernel_width,
                                    conv_filters=conv_filters, device=device)
        self.dnn_input_dim = self.conv_layer.filed_shape * self.embedding_size * conv_filters[-1]
        self.dnn = DNN(self.dnn_input_dim, dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        linear_logit = self.linear_model(X)
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict, support_dense=False)
        if len(sparse_embedding_list) == 0:
            raise ValueError("must have the embedding feature,now the embedding feature is None!")
        conv_input = concat_fun(sparse_embedding_list, axis=1)
        conv_input_concact = torch.unsqueeze(conv_input, 1)
        pooling_result = self.conv_layer(conv_input_concact)
        flatten_result = pooling_result.view(pooling_result.size(0), -1)
        dnn_output = self.dnn(flatten_result)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred
