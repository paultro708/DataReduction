from ._Reduction import _Reduction
from .ENN import ENN
from ..DataPreparation import DataPreparation
from ._NNGraph import _NNGraph
import numpy as np
from time import process_time

class ICF(_Reduction):
    """Iterative Case Filtering algorithm first apply ENN algorith to reduce noise instances, than iteratively removes other 
    instances as follows. At each iteration, created coverage and reachable subsets. If reachable is smaller than coverage for instance, it is removed.

    Args:
        _Reduction: abstract class of reduction algorithm

    Attributes:
        data (DataPreparation): DataPreparation instance with original dataset 
        red_data: reduced dataset
        red_lab: labels of reduced dataset
        max_iter (int): maximal number of iterations
        keep: array with ones, shape same as red_lab
        graph: graph with neighbours and enemies
        coverage: coverage subset for each instance
        reachable: reachable subset for each instance

    """

    def __init__(self, data: DataPreparation, max_iter: int = 3):
        """Constructor of ICF class. 

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


    def _create_cov_reach(self, index:int):
        """Function creating coverage subset. Coverage subset contains nearest neighbours with same label, limited by first enemy.
        Without itself.

        Args:
            index (int): index of instance for which creating coverage subset 

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
        """Function initialize necessary parametres for actual reduced dataset:
            keep: array with ones, shape same as red_lab
            graph: graph with neighbours and enemies
            coverage: coverage subset for each instance
            reachable: reachable subset for each instance
        """
        #init array for marking keeping instances
        self.keep = np.ones((self.red_lab.shape))
            
        #create graph to get information about neighbours and enemies for each instance 
        self.graph.create_graph(self.red_data, self.red_lab, n_enem=True)  

        #init coverage and reachable subset
        self.coverage = []
        self.reachable = []
        for i in range(len(self.red_lab)):
            self.coverage.append([])
            self.reachable.append([])

        for idx, p in enumerate(self.red_lab):
            self._create_cov_reach(idx)

    def _delete_marked(self) -> bool:
        """Function deleting instanes marked as removal. Returns the status of the operation. If it is not possible - 
        after deleting in the dataset would not remain representation of all classes, returns False.

        Returns:
            bool: the status of the deleting operation
        """
        #get indexes where red_data is marked as removal and delete them
        to_remove = np.where(self.keep == 1)[0]
        removed_l = np.delete(self.red_lab, to_remove, axis=0)
        #delete only if still will be instances of all classes in reduced dataset
        if(len(set(removed_l))==self.data.n_classes):
            self.red_data = np.delete(self.red_data, to_remove, axis=0)
            self.red_lab = removed_l
            return True
        else: 
            return False
    
    def reduce_instances(self, return_time = False, return_n_iter = False):
        print('Reducing the dataset using the ICF algorithm...')
        #start time measurement
        start = process_time()

        #apply ENN algorithm first
        enn = ENN(self.data)
        enn.reduce_instances()

        #initialize reduced data and labels
        self.red_data = enn.red_data
        self.red_lab = enn.red_lab

        #normalize data
        self.red_data, self.weights = self.data.normalize(self.red_data)

        #create graph to get information about neighbours and enemies
        self.graph = _NNGraph()

        #initialize necessary parametres
        self._init_params()

        #main part of algorithm - loop
        progress = True
        iteration = 0
        while (progress and iteration < self.max_iter):
            progress = False
            for idx, p in enumerate(self.red_lab):
                cov = self.coverage[idx]
                reach = self.reachable[idx]
                if (len(reach) > len(cov)):
                    #mark the instance as removal
                    self.keep[idx] = 0 
                    progress = True
            
            #delete marked instances or break the loop if it is not possible 
            if self._delete_marked() is False:
                break

            #if there will be another iteration, reinitialize all parametres: keep, coverage and reachable
            if (progress and iteration < self.max_iter):
                self._init_params()
            iteration +=1

        #reverse normaize
        self.red_data = self.data.reverse_normalize(self.red_data, self.weights)

        #end time measurement
        end = process_time()
        time = end - start

        #return apriopriate value/values
        if return_time and return_n_iter:
            return time, iteration+1
        elif return_time:
            return time
        elif return_n_iter:
            return iteration+1