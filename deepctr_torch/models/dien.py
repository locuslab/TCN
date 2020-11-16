"""
Author:
    Ze Wang, wangze0801@126.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .basemodel import BaseModel
from ..layers import *
from ..inputs import *


class DIEN(BaseModel):
    """Instantiates the Deep Interest Evolution Network architecture.

       :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
       :param history_feature_list: list,to indicate  sequence sparse field
       :param gru_type: str,can be GRU AIGRU AUGRU AGRU
       :param use_negsampling: bool, whether or not use negtive sampling
       :param alpha: float ,weight of auxiliary_loss
       :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
       :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
       :param dnn_activation: Activation function to use in DNN
       :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
       :param att_activation: Activation function to use in attention net
       :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
       :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
       :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
       :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
       :param init_std: float,to use as the initialize std of embedding vector
       :param seed: integer ,to use as random seed.
       :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
       :param device: str, ``"cpu"`` or ``"cuda:0"``
       :return: A PyTorch model instance.
    """

    def __init__(self,
                 dnn_feature_columns, history_feature_list,
                 gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu',
                 att_hidden_units=(64, 16), att_activation="relu", att_weight_normalization=True,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
                 device='cpu'):
        super(DIEN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std, seed=seed, task=task, device=device)

        self.item_features = history_feature_list
        self.use_negsampling = use_negsampling
        self.alpha = alpha
        self._split_columns()

        # structure: embedding layer -> interest extractor layer -> interest evolution layer -> DNN layer -> out

        # embedding layer
        # inherit -> self.embedding_dict
        input_size = self._compute_interest_dim()
        # interest extractor layer
        self.interest_extractor = InterestExtractor(input_size=input_size, use_neg=use_negsampling, init_std=init_std)
        # interest evolution layer
        self.interest_evolution = InterestEvolving(
            input_size=input_size,
            gru_type=gru_type,
            use_neg=use_negsampling,
            init_std=init_std,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization)
        # DNN layer
        dnn_input_size = self._compute_dnn_dim() + input_size
        self.dnn = DNN(dnn_input_size, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn,
                       init_std=init_std, seed=seed)
        self.linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        # prediction layer
        # inherit -> self.out

        # init
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, X):
        # [B, H] , [B, T, H], [B, T, H] , [B]
        query_emb, keys_emb, neg_keys_emb, keys_length = self._get_emb(X)
        # [b, T, H],  [1]  (b<H)
        masked_interest, aux_loss = self.interest_extractor(keys_emb, keys_length, neg_keys_emb)
        self.add_auxiliary_loss(aux_loss, self.alpha)
        # [B, H]
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)
        # [B, H2]
        deep_input_emb = self._get_deep_input_emb(X)
        deep_input_emb = concat_fun([hist, deep_input_emb])
        dense_value_list = get_dense_input(X, self.feature_index, self.dense_feature_columns)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        # [B, 1]
        output = self.linear(self.dnn(dnn_input))
        y_pred = self.out(output)
        return y_pred

    def _get_emb(self, X):
        # history feature columns : pos, neg
        history_feature_columns = []
        neg_history_feature_columns = []
        sparse_varlen_feature_columns = []
        history_fc_names = list(map(lambda x: "hist_" + x, self.item_features))
        neg_history_fc_names = list(map(lambda x: "neg_" + x, history_fc_names))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in history_fc_names:
                history_feature_columns.append(fc)
            elif feature_name in neg_history_fc_names:
                neg_history_feature_columns.append(fc)
            else:
                sparse_varlen_feature_columns.append(fc)

        # convert input to emb
        features = self.feature_index
        query_emb_list = embedding_lookup(X, self.embedding_dict, features, self.sparse_feature_columns,
                                          return_feat_list=self.item_features, to_list=True)
        # [batch_size, dim]
        query_emb = torch.squeeze(concat_fun(query_emb_list), 1)

        keys_emb_list = embedding_lookup(X, self.embedding_dict, features, history_feature_columns,
                                         return_feat_list=history_fc_names, to_list=True)
        # [batch_size, max_len, dim]
        keys_emb = concat_fun(keys_emb_list)

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        # [batch_size]
        keys_length = torch.squeeze(maxlen_lookup(X, features, keys_length_feature_name), 1)

        if self.use_negsampling:
            neg_keys_emb_list = embedding_lookup(X, self.embedding_dict, features, neg_history_feature_columns,
                                                 return_feat_list=neg_history_fc_names, to_list=True)
            neg_keys_emb = concat_fun(neg_keys_emb_list)
        else:
            neg_keys_emb = None

        return query_emb, keys_emb, neg_keys_emb, keys_length

    def _split_columns(self):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns)) if len(
            self.dnn_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.dnn_feature_columns)) if len(
            self.dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat),
                   self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.item_features:
                interest_dim += feat.embedding_dim
        return interest_dim

    def _compute_dnn_dim(self):
        dnn_input_dim = 0
        for fc in self.sparse_feature_columns:
            dnn_input_dim += fc.embedding_dim
        for fc in self.dense_feature_columns:
            dnn_input_dim += fc.dimension
        return dnn_input_dim

    def _get_deep_input_emb(self, X):
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              mask_feat_list=self.item_features, to_list=True)
        dnn_input_emb = concat_fun(dnn_input_emb_list)
        return dnn_input_emb.squeeze(1)


class InterestExtractor(nn.Module):
    def __init__(self, input_size, use_neg=False, init_std=0.001, device='cpu'):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], 'sigmoid', init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(self, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

        Returns
        -------
        masked_interests: 2D tensor, [b, H]
        aux_loss: [1]
        """
        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)
        aux_loss = torch.zeros((1,), device=keys.device)

        # create zero mask for keys_length, to make sure 'pack_padded_sequence' safe
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]

        # batch_size validation check
        if masked_keys_length.shape[0] == 0:
            return zero_outputs,

        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)

        packed_keys = pack_padded_sequence(masked_keys, lengths=masked_keys_length, batch_first=True,
                                           enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                           total_length=max_length)

        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(
                interests[:, :-1, :],
                masked_keys[:, 1:, :],
                masked_neg_keys[:, 1:, :],
                masked_keys_length - 1)

        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        # keys_length >= 1
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)

        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length,
                                                                                       embedding_size)
        batch_size = states.size()[0]

        mask = (torch.arange(max_seq_length, device=states.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1)).float()

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(click_input.view(
            batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(noclick_input.view(
            batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self,
                 input_size,
                 gru_type='GRU',
                 use_neg=False,
                 init_std=0.001,
                 att_hidden_size=(64, 16),
                 att_activation='sigmoid',
                 att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size,
                                                 gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, hidden_size = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]

        # check batch validation
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length, batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length, batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0) # [b, H]
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [b, T]
            packed_interests = pack_padded_sequence(keys, lengths=keys_length, batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length, batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length) # [b, H]
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs
        return zero_outputs
