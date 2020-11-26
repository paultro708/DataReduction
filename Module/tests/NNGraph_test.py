import pytest
from Module.InstanceReduction.Reduction._NNGraph import _NNGraph
from Module.InstanceReduction.DataPreparation import DataPreparation
#test static method group_neigh_enemies

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

#neigh: [4,6,5]
#enemy: [2,1,0,3,7,9,8]
# @pytest.mark.parametrize('labels, index, sort', [(labels), (index), (sort)])
def test_group_n_e_ok(data_iris, labels, index, sort):
    """For parametres:
    labels = [0,0,0,0,1,1,1,2,2,2]
    index = 4
    sort = [4,2,1,0,3,6,7,9,8,5]
    Should returm:
    #neigh: [4,6,5]
    #enemy: [2,1,0,3,7,9,8]
    """
    gr = _NNGraph()
    assert gr.group_neigh_enemies(labels, index, sort) == ([4,6,5],[2,1,0,3,7,9,8])

def test_group_n_e():
    
    