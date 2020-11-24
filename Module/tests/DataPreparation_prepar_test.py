from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest
# class DataPreparation_test:


@pytest.mark.parametrize("dataset, n_classes", [('iris', 3), ('pendigits', 10), ('letter', 26)])
def test_n_classes(dataset, n_classes):
    """
    Check if number of classes in prepared data is apriopriate
    """
    n_cl = DataPreparation(dataset).n_classes
    assert n_cl == n_classes