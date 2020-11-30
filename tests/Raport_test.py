import os
import pytest
from tests.const import classifiers
from InstanceReduction.Raport import Raport
import shutil

def test_raport_create(raport_iris):
    assert len(raport_iris.original.data_all) == 150

def test_print_raport_allclf(raport_iris):
    raport_iris.print_raport(show_cf_matrix=False, save_cf_matrix = True)
    path = os.path.join(os.getcwd(), 'plots')
    assert len(os.listdir(path)) == len(classifiers)*2
    shutil.rmtree(path)

# def test_print_raport_before_reduction(reduction_alg_iris, data_prepar_iris):
#     r = Raport(data_prepar_iris, reduction_alg_iris.red_data, reduction_alg_iris.red_lab)
#     with pytest.raises(Exception, match="Cannot create"):
#         r.print_raport(show_cf_matrix=False)

@pytest.mark.parametrize('clf', classifiers)
def test_print_raport_onecfl(raport_iris, clf):
    raport_iris.print_raport(show_cf_matrix=False, save_cf_matrix = True, c_type=clf)
    path = os.path.join(os.getcwd(), 'plots')
    assert len(os.listdir(path)) == 2
    shutil.rmtree(path)