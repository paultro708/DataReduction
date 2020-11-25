from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest
# class DataPreparation_test:



# @pytest.fixture(params=[
#     {'iris': [150, 3], 
#     'pendigits': [10992, 10], 
#     'letter': [20000, 26]}
# ])
names = ['iris', 'pendigits', 'letter']
len_all = [150, 10992, 20000]
n_classes = [3, 10, 26]
n_features = [4, 16, 16]



def tuples(x,y):
    """Function merging two list into tuples

    Args:
        x (list): list with same shape
        y (list): list with same shape

    Returns:
        list of tuples
    """
    return [(x[i], y[i]) for i in range(0, len(x))] 

# @pytest.fixture(params=['iris', 'pendigits', 'letter'])
# def dataset_names(request):
#     return request.param

# @pytest.fixture(params=[150, 10992, 20000])
# def len_data_all(request):
#     return request.param

# @pytest.fixture()
# def tuples(x,y):
#     return [(x[i], y[i]) for i in range(0, len(x))] 


@pytest.mark.parametrize("dataset, n_classes", tuples(names, n_classes))
def test_n_classes(dataset, n_classes):
    """
    Check if number of classes in prepared data is apriopriate
    """
    n_cl = DataPreparation(dataset).n_classes
    assert n_cl == n_classes


@pytest.mark.parametrize("dataset_names, lenn", tuples(names, len_all))
def test_len_data_all(dataset_names, lenn):
    d = DataPreparation(dataset_names)
    assert len(d.data_all) == lenn

