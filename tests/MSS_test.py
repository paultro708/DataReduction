import pytest
from InstanceReduction.Reduction.MSS import MSS
from InstanceReduction.DataPreparation import DataPreparation

@pytest.fixture
def labels():
    return [0,0,0,0,1,1,1,2,2,2]

@pytest.fixture
def index():
    return 4

@pytest.fixture
def sort():
    return [4,2,1,0,3,6,7,9,8,5]

@pytest.fixture
def data_iris():
    return DataPreparation('iris')

def test_group_n_e_ok(data_iris, labels, index, sort):
    """For parametres:
    labels = [0,0,0,0,1,1,1,2,2,2]
    index = 4
    sort = [4,2,1,0,3,6,7,9,8,5]
    Should returm:
    #neigh: [4,6,5]
    #enemy: [2,1,0,3,7,9,8]
    """
    mss = MSS(data_iris)
    assert mss.group_neigh_enemies(labels, index, sort) == ([4,6,5],[2,1,0,3,7,9,8])

def test_group_n_e_iderr(data_iris, labels, sort):
    """Chacking raining index error if parameter out of range
    """
    mss = MSS(data_iris)
    with pytest.raises(IndexError): 
        mss.group_neigh_enemies(labels, 15, sort) == ([4,6,5],[2,1,0,3,7,9,8])

def test_group_n_e_type(data_iris, sort):
    """Chacking raining type error
    """
    mss = MSS(data_iris)
    with pytest.raises(TypeError): 
        mss.group_neigh_enemies('x43', 1, 'dsjhfrt') == ([4,6,5],[2,1,0,3,7,9,8])