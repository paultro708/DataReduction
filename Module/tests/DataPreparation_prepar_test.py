from Module.InstanceReduction.DataPreparation import DataPreparation
import pytest
from Module.tests.const import names, len_all, tuples, n_classes, n_features
# class DataPreparation_test:



# @pytest.fixture(params=[
#     {'iris': [150, 3], 
#     'pendigits': [10992, 10], 
#     'letter': [20000, 26]}
# ])
@pytest.fixture(params=names)
def datasets(request):
    return DataPreparation(request.param)

@pytest.fixture(params=len_all)
def len_data_all(request):
    return request.param

@pytest.fixture(params = ["Module\\tests\\test_csv\\only_cols.csv", "Module\\tests\\test_csv\\only_cols.csv"])
def empty_csv(request):
    return request.param

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

@pytest.mark.parametrize("dataset_names, n_features", tuples(names, n_features))
def test_len_data_all(dataset_names, n_features):
    """ Check if number of fetutures is apriopriate
    """
    d = DataPreparation(dataset_names)
    assert len(d.features) == n_features

def test_normalization(datasets):
    """Check if data is normalized
    """
    assert -1 <= datasets.data_all.all() <=1

# @pytest.mark.parametrize("dataset_names, lenn", tuples(names, len_all))

def test_train_test_split(datasets):
    """Check if dataset is splitted correctly :
    sum of train and test should be original dataset and test size should be about 0.3 of original"""
    assert len(datasets.data_label_test) + len(datasets.data_label_train) == len(datasets.data_all)
    assert len(datasets.data_label_test)/len(datasets.data_label) < 0.31

#tests raising exceptions
def test_empty(empty_csv):
    with pytest.raises(Exception, match="empty"):
        DataPreparation(filepath=empty_csv, sep = ";")

def test_not_enough_cols():
    with pytest.raises(Exception, match="minimum 2 columns"):
        DataPreparation(filepath="Module\\tests\\test_csv\\one_col.csv", sep = ";")

def test_missing_values():
    with pytest.raises(Exception, match="Nan"):
        DataPreparation(filepath="Module\\tests\\test_csv\\missed.csv", sep = ";")

def test_non_numeric():
    with pytest.raises(Exception, match="non numeric"):
        DataPreparation(filepath="Module\\tests\\test_csv\\non_numeric.csv", sep = ";")