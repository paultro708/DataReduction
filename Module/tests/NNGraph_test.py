import pytest
import numpy as np
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
def test_group_n_e_ok(labels, index, sort):
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

wrong_types_lis = [0, (-3.4), -1, 'vf', False, None]
@pytest.mark.parametrize('wrong_types', wrong_types_lis)
def test_group_n_e_lis(labels, index, sort, wrong_types):
    """Checking raising TypeError in arrays/lists
    """
    gr = _NNGraph()
    with pytest.raises(TypeError):
        gr.group_neigh_enemies(wrong_types, index, sort)
    with pytest.raises(TypeError):
        gr.group_neigh_enemies(labels, index, wrong_types)


wrong_types_id = [ (-3.4), 'vf', [], False, None]
@pytest.mark.parametrize('wrong_types', wrong_types_id)
def test_group_n_e_id(labels, sort, wrong_types):
    """Checking raising TypeError in index
    """
    gr = _NNGraph()
    with pytest.raises(TypeError):
        gr.group_neigh_enemies(labels, wrong_types, sort)


@pytest.mark.parametrize('wrong_val', [[], np.array([]), np.zeros((2,1))])
def test_group_n_e_lis(labels, index, sort, wrong_val):
    """Checking raising TypeError in arrays/lists
    """
    gr = _NNGraph()
    with pytest.raises(ValueError):
        gr.group_neigh_enemies(wrong_val, index, sort)
    with pytest.raises(ValueError):
        gr.group_neigh_enemies(labels, index, wrong_val)