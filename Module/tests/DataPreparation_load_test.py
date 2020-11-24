from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest
# class DataPreparation_test:
def test_load_named():
    
    named = DataPreparation('iris')
    assert named.n_classes == 3  

def test_load_named_wrong():
    with pytest.raises(Exception, match='not found'):
        named = DataPreparation('pedigits')

def test_load_named():
    named = DataPreparation('pendigits')
    assert named.n_classes == 10 


