import os
import pytest
from Module.tests.const import classifiers
from Module.InstanceReduction.Raport import Raport

def test_raport_create(raport_iris):
    assert len(raport_iris.original.data_all) == 150

def test_print_raport(raport_iris):
    raport_iris.print_raport(show_cf_matrix=False, save_cf_matrix = True)
    path = os.path.join(os.getcwd(), 'plots')
    assert len(os.listdir(path)) == len(classifiers)*2

def test_print_raport_before_reduction(reduction_alg_iris, data_prepar_iris):
    r = Raport(data_prepar_iris, reduction_alg_iris.red_data, reduction_alg_iris.red_lab)
    with pytest.raises(Exception, match="Cannot create"):
        r.print_raport(show_cf_matrix=False)