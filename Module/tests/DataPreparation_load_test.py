from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest
from Module.tests.const import names, wrong_names, wrong_name_types, wrong_sep_types
# class DataPreparation_test:

def test_load_empty():
    """Check raising Exception when no parameters select
    """
    with pytest.raises(Exception, match="without name or filepath"):
        d = DataPreparation()

#tests with named dataset


def test_load_named(data):
    
    assert data.data_all is not None
    assert data.dataset is not None


@pytest.mark.parametrize("wrong_names", wrong_names)
def test_load_named_wrong_name(wrong_names):
    with pytest.raises(ValueError):
        wg = DataPreparation(wrong_names)

@pytest.mark.parametrize("wrong_name_types", wrong_name_types)
def test_load_named_wrong_name_types(wrong_name_types):
    with pytest.raises(TypeError):
        wg = DataPreparation(wrong_name_types)

def test_load_named_more():
    """Check loading data with apriopriate name of dataset and different args. Should take into account dataset name.
    Even if there is selected apriopriate filepath.
    """
    d= DataPreparation('iris', "D:\Studia\inz\Repos\DataReduction\Module\InstanceReduction\datasets_csv\pendigits.csv")
    assert len(d.data_all) == 150


#test with filepath
def test_load_file_absolute():
    """Check if load with absolute filepath with apriopriate other parameters (no missing values, existing class name etc.)
    """
    d = DataPreparation(filepath="D:\Studia\inz\Repos\DataReduction\Module\InstanceReduction\datasets_csv\iris.csv")
    assert len(d.dataset) == 150


def test_load_file_rel_wd():
    """Check if load with absolute filepath with apriopriate other parameters (no missing values, existing class name etc.)
    """
    d = DataPreparation(filepath="Module\InstanceReduction\datasets_csv\iris.csv")
    assert len(d.dataset) == 150

def test_load_file_not_found():
    """Check raising FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        d = DataPreparation(filepath="Module\InstanceReduction\datasets_csv\notexisting.csv")

def test_load_file_os():
    """Check raising OSError
    """
    with pytest.raises(OSError):
        d = DataPreparation(filepath="Module\InstanceReduction")


def test_load_file_notcsv():
    """Check raising ValueError
    """
    with pytest.raises(ValueError):
        d = DataPreparation(filepath="D:\Studia\inz\download.pdf")


@pytest.mark.parametrize("wrong_sep_types", wrong_sep_types)
def test_load_file_wrong_sep(wrong_sep_types):
    """Check raising TypeError when wrong separator
    """
    with pytest.raises(TypeError):
        d = DataPreparation(filepath="Module\InstanceReduction\datasets_csv\iris.csv", sep = wrong_sep_types)