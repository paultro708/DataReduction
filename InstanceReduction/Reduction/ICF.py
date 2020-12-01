from ._Reduction import _Reduction
from .ENN import ENN
from ..DataPreparation import DataPreparation
from ._NNGraph import _NNGraph
import numpy as np
from time import process_time

class ICF(_Reduction):

    def __init__(self, data: DataPreparation, max_iter: int = 3):
        """Constructor of ICF class. Iterative Case Filtering algorithm first apply ENN algorith to reduce noise instances, than
        iteratively removes other instances as follows. At each iteration, created coverage and reachable subsets. If reachable is 
        smaller than coverage for instance, it is removed.

        Args:
            data (DataPreparation): instance of :DataPreparation class containing prepared dataset for ENN alhorithm application
            max_iter (int): maximal number of performed interations
            
        Raises:
            TypeError: when given parameter does not have apriopriate type
        """
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        self.data = data
        self.red_data = []
        self.red_lab = []
        self.max_iter = max_iter

        if type(max_iter) != int:
            raise TypeError('max_iter atribute must be integer value')
        elif max_iter < 1:
            raise ValueError('max_iter atribute must have value not less than 1')


    def create_cov_reach(self, index:int):
        """Function creating coverage subset. Coverage subset contains nearest neighbours with same label, limited by first enemy.
        Without itself.

        Args:
            index (int): index of instance creating coverage subset for 

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

    def _init_params(self):
        #init arrays for keeping instances
        self.keep = np.ones((self.red_lab.shape))
            
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
    
    def reduce_instances(self, return_time = False, return_n_iter = False):
        print('Reducing the dataset using the ICF algorithm...')
        #start time measurement
        start = process_time()

        #apply ENN algorithm first
        enn = ENN(self.data)
        enn.reduce_instances()

        #init reduced data and labels
        self.red_data = enn.red_data
        self.red_lab = enn.red_lab

        #create graph to get information about neighbours and enemies
        self.graph = _NNGraph()
        self._init_params()

        #do algorithm
        progress = True
        iteration = 0
        while (progress and iteration < self.max_iter):
            progress = False
            for idx, p in enumerate(self.red_lab):
                cov = self.coverage[idx]
                reach = self.reachable[idx]
                if (len(reach) > len(cov)):
                    self.keep[idx] = 0
                    progress = True
            
            #get indexes where red_data is marked to remove and delete them
            to_remove = np.where(self.keep == 1)[0]
            removed_l = np.delete(self.red_lab, to_remove, axis=0)
            #remove only if still will be all classes
            if(len(set(removed_l))==self.data.n_classes):
                self.red_data = np.delete(self.red_data, to_remove, axis=0)
                self.red_lab = removed_l
            else: 
                break

            #reinit all params: keep, coverage and reachable
            #if it would be next iter
            if (progress and iteration < self.max_iter):
                self._init_params()
            iteration +=1

        #end time measurement
        end = process_time()
        time = end - start

        if return_time and return_n_iter:
            return time, iteration+1
        elif return_time:
            return time
        elif return_n_iter:
            return iteration+1