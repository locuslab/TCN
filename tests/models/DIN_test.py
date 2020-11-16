# -*- coding: utf-8 -*-
import numpy as np

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.din import DIN
from ..utils import check_model, get_device


def get_xy_fd(hash_flag=False):
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

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1, 0])
    return x, y, feature_columns, behavior_feature_list


def test_DIN():
    model_name = "DIN"

    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list, dnn_dropout=0.5, device=get_device())

    check_model(model, model_name, x, y)  # only have 3 train data so we set validation ratio at 0


if __name__ == "__main__":
    pass
