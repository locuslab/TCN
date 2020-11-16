# -*- coding:utf-8 -*-
"""
Author:
    Junyi Huo
Reference:
    [1] Yang Y, Xu B, Shen F, et al. Operation-aware Neural Networks for User Response Prediction[J]. arXiv preprint arXiv:1904.12579, 2019. （https://arxiv.org/pdf/1904.12579）
"""

from .basemodel import *
from ..inputs import combined_dnn_input
from ..layers import DNN


class Interac(nn.Module):
    def __init__(self, first_size, second_size, emb_size, init_std, sparse=False):
        super(Interac, self).__init__()
        self.emb1 = nn.Embedding(first_size, emb_size, sparse=sparse)
        self.emb2 = nn.Embedding(second_size, emb_size, sparse=sparse)
        self.__init_weight(init_std)

    def __init_weight(self, init_std):
        nn.init.normal_(self.emb1.weight, mean=0, std=init_std)

    def forward(self, first, second):
        """
        input:
            x batch_size * 2
        output:
            y batch_size * emb_size
        """
        first_emb = self.emb1(first)
        second_emb = self.emb2(second)
        y = first_emb * second_emb  # core code
        return y


class ONN(BaseModel):
    """Instantiates the Operation-aware Neural Networks  architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param use_bn: bool,whether use bn after ffm out or not
    :param reduce_sum: bool,whether apply reduce_sum on cross vector
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0,
                 dnn_dropout=0, init_std=0.0001, seed=1024, dnn_use_bn=False, dnn_activation='relu',
                 task='binary', device='cpu'):
        super(ONN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device)

        # second order part
        embedding_size = self.embedding_size
        self.second_order_embedding_dict = self.__create_second_order_embedding_matrix(
            dnn_feature_columns, embedding_size=embedding_size, sparse=False).to(device)

        # add regularization for second_order_embedding
        self.add_regularization_weight(
            self.second_order_embedding_dict.parameters(), l2_reg_embedding)

        dim = self.__compute_nffm_dnn_dim(
            feature_columns=dnn_feature_columns, embedding_size=embedding_size)
        self.dnn = DNN(inputs_dim=dim,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn,
                       dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2_reg_dnn)
        self.to(device)

    def __compute_nffm_dnn_dim(self, feature_columns, embedding_size):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        return int(len(sparse_feature_columns) * (len(sparse_feature_columns) - 1) / 2 * embedding_size +
                   sum(map(lambda x: x.dimension, dense_feature_columns)))

    def __input_from_second_order_column(self, X, feature_columns, second_order_embedding_dict):
        '''
        :param X: same as input_from_feature_columns
        :param feature_columns: same as input_from_feature_columns
        :param second_order_embedding_dict: ex: {'A1+A2': Interac model} created by function create_second_order_embedding_matrix
        :return:
        '''
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        second_order_embedding_list = []
        for first_index in range(len(sparse_feature_columns) - 1):
            for second_index in range(first_index + 1, len(sparse_feature_columns)):
                first_name = sparse_feature_columns[first_index].embedding_name
                second_name = sparse_feature_columns[second_index].embedding_name
                second_order_embedding_list.append(
                    second_order_embedding_dict[first_name + "+" + second_name](
                        X[:, self.feature_index[first_name][0]
                             :self.feature_index[first_name][1]].long(),
                        X[:, self.feature_index[second_name][0]
                             :self.feature_index[second_name][1]].long()
                    )
                )
        return second_order_embedding_list

    def __create_second_order_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        temp_dict = {}
        for first_index in range(len(sparse_feature_columns) - 1):
            for second_index in range(first_index + 1, len(sparse_feature_columns)):
                first_name = sparse_feature_columns[first_index].embedding_name
                second_name = sparse_feature_columns[second_index].embedding_name
                temp_dict[first_name + "+" + second_name] = Interac(sparse_feature_columns[first_index].vocabulary_size,
                                                                    sparse_feature_columns[
                                                                        second_index].vocabulary_size,
                                                                    emb_size=embedding_size,
                                                                    init_std=init_std,
                                                                    sparse=sparse)
        return nn.ModuleDict(temp_dict)

    def forward(self, X):

        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                              self.embedding_dict)
        linear_logit = self.linear_model(X)
        spare_second_order_embedding_list = self.__input_from_second_order_column(X, self.dnn_feature_columns,
                                                                                  self.second_order_embedding_dict)
        dnn_input = combined_dnn_input(
            spare_second_order_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        if len(self.dnn_feature_columns) > 0:
            final_logit = dnn_logit + linear_logit
        else:
            final_logit = linear_logit

        y_pred = self.out(final_logit)

        return y_pred
