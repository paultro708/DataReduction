from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.ENN import ENN
from InstanceReduction.Reduction.DROP1 import DROP1
from InstanceReduction.Reduction.PCS import PCS
from InstanceReduction.Reduction.MSS import MSS
from InstanceReduction.Reduction.ICF import ICF

data = DataPreparation('iris')#filepath="D:\Studia\inz\datasets_csv\glass.csv", class_col='Type')#'iris')
alg = DROP1(data, 3)
print(alg.reduce_instances(return_time=True))

from InstanceReduction.Raport import Raport

rap = Raport(data, alg.red_data, alg.red_lab)
# rap.draw_plots('Aattr', 'Battr', show = True, save = True) #,'sepallength','sepalwidth' 'Aattr', 'Battr' 'width', 'high'

# rap.draw_plots('mcv','drinks', show = True, save = True) #liver
rap.draw_plots('sepallength','sepalwidth', show = True, save = True)
rap.print_raport(c_type='knn', show_cf_matrix = False, save_cf_matrix= True)#, norm=True) #c_type = 'all',show_cf_matrix = True, path ='', save_plots = False)

