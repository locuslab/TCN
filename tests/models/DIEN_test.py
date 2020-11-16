import numpy as np
import pytest
import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models.dien import InterestEvolving, DIEN
from ..utils import check_model, get_device


@pytest.mark.parametrize(
    'gru_type',
    ["AIGRU", "AUGRU", "AGRU", "GRU"]
)
def test_InterestEvolving(gru_type):
    interest_evolution = InterestEvolving(
        input_size=3,
        gru_type=gru_type,
        use_neg=False)

    query = torch.tensor([[1, 1, 1], [0.1, 0.2, 0.3]], dtype=torch.float)

    keys = torch.tensor([
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3], [1, 2, 3], [0.4, 0.2, 1], [0.5, 0.5, 0.5]]
    ], dtype=torch.float)

    keys_length = torch.tensor([3, 4])

    output = interest_evolution(query, keys, keys_length)

    assert output.size()[0] == 2
    assert output.size()[1] == 3


def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 4, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'),
                         maxlen=4,
                         length_name="seq_length")]

    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2, 3])
    gender = np.array([0, 1, 0, 1])
    item_id = np.array([1, 2, 3, 2])  # 0 is mask value
    cate_id = np.array([1, 2, 1, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3, 0.2])

    hist_item_id = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])

    behavior_length = np.array([3, 3, 2, 2])

    feature_dict = {'user': uid, 'gender': gender, 'item_id': item_id, 'cate_id': cate_id,
                    'hist_item_id': hist_item_id, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(
                SparseFeat('neg_hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'),
                maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1, 0])
    return x, y, feature_columns, behavior_feature_list


@pytest.mark.parametrize(
    'gru_type,use_neg',
    [("AIGRU", True), ("AIGRU", False),
     ("AUGRU", True), ("AUGRU", False),
     ("AGRU", True), ("AGRU", False),
     ("GRU", True), ("GRU", False)]
)
def test_DIEN(gru_type, use_neg):
    model_name = "DIEN_" + gru_type

    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=use_neg)

    model = DIEN(feature_columns, behavior_feature_list, gru_type=gru_type, use_negsampling=use_neg,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, device=get_device())

    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
