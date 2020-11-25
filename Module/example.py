from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.ENN import ENN
from InstanceReduction.Reduction.DROP1 import DROP1
from InstanceReduction.Reduction.PCS import PCS

data = DataPreparation(False)
data.prepare_dataset()
enn = PCS(data, 7)
enn.reduce_instances(return_time=True)

from InstanceReduction.Raport import Raport

rap = Raport(data, enn.red_data, enn.red_lab)
rap.print_raport()