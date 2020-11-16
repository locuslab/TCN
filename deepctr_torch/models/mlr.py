# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
    Weichen Shen,wcshen1994@163.com
Reference:
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""
import torch
import torch.nn as nn

from .basemodel import Linear, BaseModel
from ..inputs import build_input_features
from ..layers import PredictionLayer


class MLR(BaseModel):
    """Instantiates the Mixed Logistic Regression/Piece-wise Linear Model.

    :param region_feature_columns: An iterable containing all the features used by region part of the model.
    :param base_feature_columns: An iterable containing all the features used by base part of the model.
    :param region_num: integer > 1,indicate the piece number
    :param l2_reg_linear: float. L2 regularizer strength applied to weight
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param bias_feature_columns: An iterable containing all the features used by bias part of the model.
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """

    def __init__(self, region_feature_columns, base_feature_columns=None, bias_feature_columns=None,
                 region_num=4, l2_reg_linear=1e-5, init_std=0.0001, seed=1024, task='binary', device='cpu'
                 ):
        super(MLR, self).__init__(region_feature_columns, region_feature_columns, task=task, device=device)

        if region_num <= 1:
            raise ValueError("region_num must > 1")

        self.l2_reg_linear = l2_reg_linear
        self.init_std = init_std
        self.seed = seed
        self.device = device

        self.region_num = region_num
        self.region_feature_columns = region_feature_columns
        self.base_feature_columns = base_feature_columns
        self.bias_feature_columns = bias_feature_columns

        if base_feature_columns is None or len(base_feature_columns) == 0:
            self.base_feature_columns = region_feature_columns

        if bias_feature_columns is None:
            self.bias_feature_columns = []

        self.feature_index = build_input_features(
            self.region_feature_columns + self.base_feature_columns + self.bias_feature_columns)

        self.region_linear_model = nn.ModuleList([Linear(
            self.region_feature_columns, self.feature_index, self.init_std, self.device) for i in
            range(self.region_num)])

        self.base_linear_model = nn.ModuleList([Linear(
            self.base_feature_columns, self.feature_index, self.init_std, self.device) for i in range(self.region_num)])

        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            self.bias_model = nn.Sequential(
                Linear(self.bias_feature_columns, self.feature_index,
                       self.init_std, self.device),
                PredictionLayer(task='binary', use_bias=False))

        self.prediction_layer = PredictionLayer(task=task, use_bias=False)

        self.to(self.device)

    def get_region_score(self, inputs, region_number):
        region_logit = torch.cat([self.region_linear_model[i](
            inputs) for i in range(region_number)], dim=-1)
        region_score = nn.Softmax(dim=-1)(region_logit)
        return region_score

    def get_learner_score(self, inputs, region_number):
        learner_score = self.prediction_layer(torch.cat(
            [self.region_linear_model[i](inputs) for i in range(region_number)], dim=-1))
        return learner_score

    def forward(self, X):

        region_score = self.get_region_score(X, self.region_num)
        learner_score = self.get_learner_score(X, self.region_num)

        final_logit = torch.sum(
            region_score * learner_score, dim=-1, keepdim=True)

        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            bias_score = self.bias_model(X)
            final_logit = final_logit * bias_score
        return final_logit
