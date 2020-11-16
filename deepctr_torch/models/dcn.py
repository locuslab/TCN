# -*- coding:utf-8 -*-
"""
Author:
    chen_kkkk, bgasdo36977@gmail.com

    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)

    [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import CrossNet, DNN


class DCN(BaseModel):
    """Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
    and DCN-M (parameterization='matrix').

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_cross=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu'):

        super(DCN, self).__init__(linear_feature_columns=linear_feature_columns,
                                  dnn_feature_columns=dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       init_std=init_std, device=device)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(
            device)
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns),
                                 layer_num=cross_num, parameterization=cross_parameterization, device=device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2_reg_linear)
        self.add_regularization_weight(self.crossnet.kernels, l2_reg_cross)
        self.to(device)

    def forward(self, X):

        logit = self.linear_model(X)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:  # Error
            pass
        y_pred = self.out(logit)
        return y_pred
