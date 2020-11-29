import pytest
from Module.InstanceReduction.DataPreparation import DataPreparation
from Module.tests.const import tuples, names, to_large_k

# def test_reduction_create(data, reduction_alg):
#     assert len(reduction_alg.red_lab) == len(data.data_label_train)
#     assert reduction_alg.red_data.shape == data.data_all_train.shape

@pytest.fixture
def data_preparation_ok():
    return DataPreparation('iris')

def test_reducing(data_preparation_ok, reduction_alg_iris):
    d = data_preparation_ok
    reduction_alg_iris.reduce_instances()
    assert len(reduction_alg_iris.red_lab) < len(d.data_label_train)
    assert reduction_alg_iris.red_data.shape[0] < d.data_all_train.shape[0]

def test_reducing_time(data_preparation_ok, reduction_alg_iris):
    d = data_preparation_ok
    time = reduction_alg_iris.reduce_instances(True)
    assert len(reduction_alg_iris.red_lab) < len(d.data_label_train)
    assert reduction_alg_iris.red_data.shape[0] < d.data_all_train.shape[0]
    assert time > 0

@pytest.mark.parametrize('data', [1, (-1.45), 'str', [], False, None])
def test_create_wrong_data(reduction_alg_names, data):
    """ Check raising exception when init without DataPreparation instance"""
    with pytest.raises(Exception , match='DataPreparation'):
        r = reduction_alg_names(data)

@pytest.mark.parametrize('wrong_k_type', [(-1.45), {'x' : 2}, 'str', [], False, None])
def test_create_wrong_kr(reduction_alg_names, data_preparation_ok, wrong_k_type):
    """ Check raising exception when init with wrong type of k parameter"""
    with pytest.raises(TypeError):
        r = reduction_alg_names(data_preparation_ok, wrong_k_type)

@pytest.mark.parametrize('dataset_name, wrong_k_value', tuples(names, to_large_k))
def test_create_too_small_kr(reduction_alg_names, dataset_name,wrong_k_value):
    """ Check raising exception when init with wrong value of k parameter"""
    d = DataPreparation(dataset_name)
    with pytest.raises(ValueError):
        r = reduction_alg_names(d, wrong_k_value)

