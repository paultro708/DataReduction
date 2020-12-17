from InstanceReduction.Raport import Raport
from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.ENN import ENN
from InstanceReduction.Reduction.DROP1 import DROP1
from InstanceReduction.Reduction.PSC import PSC
from InstanceReduction.Reduction.MSS import MSS
from InstanceReduction.Reduction.ICF import ICF

# filepath="D:\Studia\inz\datasets_csv\glass.csv", class_col='Type')#'iris')
data = DataPreparation('iris')
alg = DROP1(data)
print(alg.reduce_instances(return_time=True))


rap = Raport(data, alg.red_data, alg.red_lab)
# rap.draw_plots('Aattr', 'Battr', show = True, save = True) #,'sepallength','sepalwidth' 'Aattr', 'Battr' 'width', 'high'

# rap.draw_plots('mcv','drinks', show = True, save = True) #liver
# rap.draw_plots('sepallength','sepalwidth', show = True, save = True)

# rap.draw_plots('mcg','gvh', show = True, save = True) #yeast

# rap.draw_plots('x-box', 'y-box', show = True, save = True) #letter
# , norm=True) #c_type = 'all',show_cf_matrix = True, path ='', save_plots = False)
rap.print_raport(show_cf_matrix=False, save_cf_matrix=True)
