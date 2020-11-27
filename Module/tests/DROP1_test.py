import pytest
from Module.InstanceReduction.Reduction.DROP1 import DROP1
#from Module.tests.const

@pytest.fixture()
def label_all():
    return [0,0,0,0,1,1,1,2,2,2,2,2]
#1d
@pytest.fixture()
def data_all():
    return [4,5,7,2,5,4,5,6,3,5,6,5] 

def test_find_data_labels(label_all, data_all, data_prepar_iris):
    """Check find_data_and_labels 
    """
    d = DROP1(data_prepar_iris)
    assert d.find_data_and_labels([3,5,6], data_all,label_all) == ([2, 4, 5], [0,1,1])

def test_find_data_labels_wrong(label_all, data_all, data_prepar_iris):
    """Check find_data_and_labels with index out of range
    """
    d = DROP1(data_prepar_iris)
    with pytest.raises(IndexError):
        x = d.find_data_and_labels([3,5,6, 22], data_all,label_all)