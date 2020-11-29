import pytest
from Module.tests.const import names, basic_reduction, classifiers
from Module.InstanceReduction.DataPreparation import DataPreparation
from Module.InstanceReduction.Raport import Raport


@pytest.fixture(params=names, scope = 'module')
def data(request):
    return DataPreparation(request.param)

@pytest.fixture(params = basic_reduction, scope = 'module')
def reduction_alg(request, data):
    return request.param(data)

@pytest.fixture(params = basic_reduction, scope = 'module')
def reduction_alg_names(request):
    return request.param

@pytest.fixture(scope = 'module')
def data_prepar_iris(request):
    return DataPreparation('iris')

@pytest.fixture(params = basic_reduction, scope = 'module')
def reduction_alg_iris(request):
    return request.param(DataPreparation('iris'))

@pytest.fixture(scope='module')
def raport_iris(reduction_alg_iris, data_prepar_iris):
    reduction_alg_iris.reduce_instances()
    return Raport(data_prepar_iris, reduction_alg_iris.red_data, reduction_alg_iris.red_lab)
