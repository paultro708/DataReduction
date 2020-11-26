from Module.InstanceReduction.Reduction.ENN import ENN
from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest


@pytest.fixture()
def data():
    return DataPreparation('iris')
    

def test_create(data):
    d = data
    enn = ENN(d)
    assert enn.k == 5
    assert len(enn.red_lab) == len(d.data_label_train)
    assert enn.red_data.shape == d.data_all_train.shape

@pytest.mark.parametrize('k', [1, 2, 3, 4, 10])
def test_create_k(data, k):
    d = data
    enn = ENN(d, k)
    assert enn.k == k
    assert len(enn.red_lab) == len(d.data_label_train)
    assert enn.red_data.shape == d.data_all_train.shape

@pytest.mark.parametrize('maxk', [104])
def test_create_k(data, maxk):
    d = data
    enn = ENN(d, maxk)
    assert enn.k == maxk
    assert len(enn.red_lab) == len(d.data_label_train)
    assert enn.red_data.shape == d.data_all_train.shape


#tests wrong k:
