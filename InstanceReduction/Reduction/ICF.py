from ._Reduction import _Reduction
from .ENN import ENN
from ..DataPreparation import DataPreparation
from ._NNGraph import _NNGraph
import numpy as np
from time import process_time

class ICF(_Reduction):

    def __init__(self, data: DataPreparation, k: int=5):
        """Constructor of ENN class. Edited Nearest Neighbours algorith removes instances which label dusagrees with majority class in k nearest neighbours set.

        Args:
            data (DataPreparation): instance of :DataPreparation class containing prepared dataset for ENN alhorithm application
            k (int, optional): number of nearest neighbours takein into account during reduction. Defaults to 5.

        Raises:
            TypeError: when given parameter does not have apriopriate type
            ValueError: when given :k value is less than 1 or greater than number of instances in reducting dataset.
        """
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        self.data = data
        self.k = k
        self.red_data = []
        self.red_lab = []

        if type(k) != int:
            raise TypeError('k atribute must be integer value')
        elif k < 1:
            raise ValueError('k atribute must have value not less than 1')
        elif k >= len(self.data.data_all_train):
            raise ValueError('k atribute must have value less than number of instances in dataset')


    def create_cov_reach(self, index:int)-> list:
        """Function creating coverage subset. Coverage subset contains nearest neighbours with same label, limited by first enemy.
        Without itself.

        Args:
            index (int): index of instance creating coverage subset for

        Returns:
            list: list representing coverage subset
        """
        sort = self.graph.sort_id[index] 
        #index of nearest enemy
        id_nearest_enemy = self.graph.enemy[index][0]
        #id of nearest enemy in list of sorted ids by distance 
        place_of_nearest_enemy = np.where(sort == id_nearest_enemy)[0][0]

        #geting instances to first enemy
        self.coverage[index] = sort[1:place_of_nearest_enemy]

        #update reachable
        for idx in self.coverage[index]:
                self.reachable[idx].append(index)


    def reduce_instances(self, return_time = False):
        print('Reducing the dataset using the ICF algorithm...')

        #apply ENN algorith first
        enn = ENN(self.data)
        enn.reduce_instances()
        #init reduced data and labels
        self.red_data = enn.red_data
        self.red_lab = enn.red_lab

        #start time measurement
        start = process_time()

        #create graph for get information about neighbours and enemies
        self.graph = _NNGraph()
        self.graph.create_graph(self.red_data, self.red_lab)
        
        #do algorithm
        progress = True
        # #init arrays for keeping instances
        # keep = np.ones((self.red_data.shape))

        #init coverage and reachable
        self.coverage = []
        self.reachable = []
        for i in range(len(self.data.data_label_train)):
            self.coverage.append([])
            self.reachable.append([])

        for idx, p in enumerate(self.red_lab):
            self.create_cov_reach(idx)

        while progress:
            ###todo in def
            #init arrays for keeping instances
            keep = np.ones((self.red_lab.shape))
            
            #recreate graph
            self.graph.create_graph(self.red_data, self.red_lab)
            #init coverage and reachable subset
            self.coverage = []
            self.reachable = []
            for i in range(len(self.red_lab)):
                self.coverage.append([])
                self.reachable.append([])

            for idx, p in enumerate(self.red_lab):
                self.create_cov_reach(idx)
            ######
            progress = False
            for idx, p in enumerate(self.red_lab):
                cov = self.coverage[idx]
                reach = self.reachable[idx]
                if len(reach) > len(cov):
                    keep[idx] = 0
            
            #get indexes where red_data is marked to remove
            to_remove = np.where(keep == 1)[0]
            self.red_data = np.delete(self.red_data, to_remove, axis=0)
            self.red_lab = np.delete(self.red_lab, to_remove, axis=0)
            
        #end time measurement
        end = process_time()

        if return_time:
            return end - start