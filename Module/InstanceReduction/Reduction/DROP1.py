from ._Reduction import _Reduction
from ..DataPreparation import DataPreparation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from time import process_time
from ._NNGraph import _NNGraph

class DROP1(_Reduction):
    """
    Class representing DROP1 algorithm. It reduces especially noise instances.
    """

    def __init__(self, data: DataPreparation, k: int=3):
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        self.data = data
        self.k = k
        if type(k) != int:
            raise TypeError('k atribute must be integer value')
        elif k < 1:
            raise ValueError('k atribute must have value not less than 1')
        elif k >= len(self.data.data_label_train):
            raise ValueError('k atribute must have value less than number of instances in dataset')
        self.red_data = self.data.data_all_train
        self.red_lab = self.data.data_label_train
        self.graph = _NNGraph()
        self.graph.create_graph(self.data.data_all_train, self.data.data_label_train)

    @staticmethod
    def find_data_and_labels(tab, dataset, labelset):
        """
        Function finding data and labels for indexes in tab in dataset
        """
        #data of instances
        data_ins = []
        #labels of instances
        label_ins = []
        for i in tab:
            data_ins.append(dataset[i])
            label_ins.append(labelset[i])
        
        # if data_ins == []:
        #     return [data_ins], [label_ins]
        # else:
        return data_ins, label_ins


    def __n_classified_correct(self, index, k, without = None):
        """
        Function counting correctly classified instances
        """
        #data of instances in ANN 
        #excepted labels of instances
        data_for_pred, label_expct = self.find_data_and_labels(self.graph.assot[index], self.data.data_all_train, self.data.data_label_train) 

        #pred = knn.predict(data_for_pred)
        #count correctly classified instances in assotiates
        n = 0
        #for each assotiate of id
        for i, val in enumerate (self.graph.assot[index]):
            if self.graph.predict(val, self.data.class_dict, self.data.data_label_train, k, without) == label_expct[i]:
                n = n+1

        return n


    def reduce_instances(self, return_time = False):
        print('Dzieje sie magia DROP1')
        """
        TODO k+1 dla without i k dla with
        """
        #start time measurement
        start = process_time()

        n_instances = len(self.data.data_label_train)
        ########################
        for i, d in np.ndenumerate(self.red_data): #enumerate(self.red_data[:]):
            n_with = self.__n_classified_correct(i[0], self.k) 
            n_without = self.__n_classified_correct(i[0], self.k+1, i) 

            if (n_without >= n_with):
                """
                remove instances TODO how
                """

                self.red_data = np.delete(self.red_data, [i[0]], axis=0)
                self.red_lab = np.delete(self.red_lab, [i[0]], axis = 0)
                for j, v in np.ndenumerate(self.red_data):
                    if i[0] in self.graph.assot[j[0]]:
                        self.graph.assot[j[0]].remove(i[0]) #delete from associates

                    if i[0] in self.graph.neigh[j[0]]:
                        self.graph.neigh[j[0]].remove(i[0]) #delete from neighbours


        #end time measurement
        end = process_time()

        if return_time:
            return end - start



        


