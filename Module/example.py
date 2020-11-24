from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.ENN import ENN

data = DataPreparation()
data.prepare_dataset()
enn = ENN(data)
enn.reduce_instances(return_time=True)

from InstanceReduction.Raport import Raport

rap = Raport(data, enn.red_data, enn.red_lab)
rap.print_raport()