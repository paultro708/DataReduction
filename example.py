from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.ENN import ENN
from InstanceReduction.Reduction.DROP1 import DROP1
from InstanceReduction.Reduction.PCS import PCS
from InstanceReduction.Reduction.MSS import MSS
from InstanceReduction.Reduction.ICF import ICF

data = DataPreparation('iris')#filepath="D:\Studia\inz\datasets_csv\glass.csv", class_col='Type')#'iris')
enn = ICF(data)
print(enn.reduce_instances(return_time=True, return_n_iter=True))

from InstanceReduction.Raport import Raport

rap = Raport(data, enn.red_data, enn.red_lab)
#rap.draw_plots('Aattr', 'Battr', show = True, save = True) #,'sepallength','sepalwidth' 'Aattr', 'Battr' 'width', 'high'

rap.draw_plots('sepallength','sepalwidth', show = True, save = True)
rap.print_raport(show_cf_matrix = False, save_cf_matrix= True) #c_type = 'all',show_cf_matrix = True, path ='', save_plots = False)
