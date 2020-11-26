import pytest
from Module.tests.const import names, basic_reduction 
from Module.InstanceReduction.DataPreparation import DataPreparation


@pytest.fixture(params=names, scope = 'module')
def data(request):
    return DataPreparation(request.param)

@pytest.fixture(params = basic_reduction, scope = 'module')
def reduction_alg(request, data):
    return request.param(data)

@pytest.fixture(params = basic_reduction, scope = 'module')
def reduction_alg_names(request):
    return request.param

