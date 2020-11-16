# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, BiInteractionPooling


class NFM(BaseModel):
    """Instantiates the NFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param biout_dropout: When not ``None``, the probability we will drop out the output of BiInteractionPooling Layer.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in deep net
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, bi_dropout=0,
                 dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu'):
        super(NFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, include_sparse=False) + self.embedding_size,
                       dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2_reg_dnn)
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_logit = self.linear_model(X)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        dnn_input = combined_dnn_input([bi_out], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        logit = linear_logit + dnn_logit

        y_pred = self.out(logit)

        return y_pred
