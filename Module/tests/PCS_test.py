import pytest
from Module.InstanceReduction.Reduction.PCS import PCS
from Module.InstanceReduction.DataPreparation import DataPreparation
from Module.tests.const import len_all, tuples, names
import numpy as np


# @pytest.mark.parametrize('dat', names)
# @pytest.mark.parametrize('n_clust', [1, 2, 4, 5])
# def pcs(dat, n_clust):
#     d = DataPreparation(dat)
#     return PCS(d, n_clust)

@pytest.fixture
def tmp_pcs(request):
    d = DataPreparation('iris')
    return PCS(d)

@pytest.mark.parametrize('n_clust', [1, 2, 4, 5])
@pytest.mark.parametrize('dat, len_d', tuples(names, len_all))
def test_create_clusters(dat, len_d, n_clust):
    d = DataPreparation(dat)
    pcs = PCS(d)
    clust = pcs.create_clusters(d.data_all, n_clust)
    # poss_clus = range(0,n_clust)
    assert len(clust) == len_d
    for i in clust: assert (i in range(0,n_clust)) == True 

@pytest.mark.parametrize('n_clust', [-1, [], None, 0.56, True, 'fd'])
@pytest.mark.parametrize('dat, len_d', tuples(names, len_all))
def test_create_clusters_wrong_ncl(dat, len_d, n_clust):
    d = DataPreparation(dat)
    pcs = PCS(d)
    with pytest.raises(Exception):
        clust = pcs.create_clusters(d.data_all, n_clust)

@pytest.mark.parametrize('dat', [-1, [], None, 0.56, True, 'cv'])
def test_create_clusters_wrong_dat(dat, tmp_pcs):
    with pytest.raises(Exception):
        clust = tmp_pcs.create_clusters(dat)

#############
def test_group_id_by_clusters(tmp_pcs):
    assert tmp_pcs.group_id_by_cluster([0,3,0,0,2,2,1,1,2,3,3,1]) == {0:[0,2,3], 1:[6,7,11], 2:[4,5,8], 3:[1,9,10]}

def test_id_nearest_point(tmp_pcs):
    dat = [0, 0.4, 5, 2, 2.5]
    assert tmp_pcs.find_id_of_nearest_point(dat, [0,1,2,3,4], 0.1) == 0

def test_nearest_instance(tmp_pcs):
    dat = [0, 0.4, 5, 2, 2.5]
    assert tmp_pcs.find_nearest_instance(1, [0,1,2,3,4], dat) == 0

def test_find_majority_class(tmp_pcs):
    ncls = 3
    cls_id = [[1,2], [], [0,3,4,5]]
    assert tmp_pcs.find_majority_class(ncls, cls_id) == 2

def test_mean_point_clst(tmp_pcs):
    dat = np.array([0, 0.4, 5, 2, 2.5])
    ids = np.array([0,3])
    assert tmp_pcs.mean_point_in_cluster(dat, ids) == 1.0 #'{0:.1f}'.format(tmp_pcs.mean_point_in_cluster(dat, ids)) == '0.8'

# def test_mean_point_clst_empty(tmp_pcs):
#     arr = np.array([4,5])
#     emp = np.array([])
#     with pytest.raises(ValueError, match="empty"):
#         tmp_pcs.mean_point_in_cluster(arr, emp)
#         tmp_pcs.mean_point_in_cluster(emp, arr)

@pytest.mark.parametrize('clust, n_cls, expected', [([0,0,1,2], 3, False), ([2,2,2,2], 3, True)])
def test_check_homogenious(clust, n_cls, expected, tmp_pcs):
    assert tmp_pcs.check_homogenious(n_cls, clust) == expected

