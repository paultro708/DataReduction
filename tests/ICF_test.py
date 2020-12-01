from ..InstanceReduction.Reduction.ICF import ICF
from ..InstanceReduction.Reduction._NNGraph import _NNGraph
from ..InstanceReduction.DataPreparation import DataPreparation
import pytest
import numpy as np

@pytest.fixture()
def data_prepar_iris(request):
    return DataPreparation('iris')

@pytest.fixture()
def prepared_icf(data_prepar_iris):
    icf = ICF(data_prepar_iris)
    icf.graph = _NNGraph()
    icf.red_data, icf.red_lab = data_prepar_iris.data_all, data_prepar_iris.data_label
    icf._init_params()
    return icf

def test_default_max_iter(data_prepar_iris):
    icf = ICF(data_prepar_iris)
    assert icf.max_iter == 3

def test_reduce_max_iter(data_prepar_iris):
    icf = ICF(data_prepar_iris, 10)
    assert icf.max_iter == 10
    assert icf.reduce_instances(return_n_iter=True) <=10

def test_cov_reach(data_prepar_iris, prepared_icf):
    prepared_icf._create_cov_reach(0)
    assert len(prepared_icf.coverage[0]) > 0

def test_init_params(data_prepar_iris, prepared_icf):
    lengh = len(data_prepar_iris.data_label)
    assert len(prepared_icf.coverage) == lengh
    assert len(prepared_icf.reachable) == lengh
    assert prepared_icf.keep.shape == data_prepar_iris.data_label.shape
    assert prepared_icf.keep[0] == 1
    assert np.all(prepared_icf.keep == 1)
    assert len(prepared_icf.coverage[0]) > 0

def test_delete_marked(data_prepar_iris, prepared_icf):
    lengh = len(data_prepar_iris.data_label)
    prepared_icf.keep[0] = 0
    if prepared_icf._delete_marked() == True:    
        assert len(prepared_icf.red_data) == (lengh - 1)
        assert len(prepared_icf.red_lab) == (lengh - 1)
    else:
        assert len(prepared_icf.red_data) == lengh
        assert len(prepared_icf.red_lab) == lengh


