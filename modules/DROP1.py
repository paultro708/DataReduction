from DataPreparation import DataPreparation
from InstanceReduction import InstanceReduction
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from time import process_time

class DROP1(InstanceReduction):
    """
    Class representing DROP1 algorithm. It reduces especially noise instances.
    """

    def __init__(self, data: DataPreparation, k=3):
        self.data = data
        self.k = k
        self.red_data = self.data.data_all_train
        self.red_lab = self.data.data_label_train

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

        # #data of instances
        # data_ins = np.empty((1,len(tab)))
        # #labels of instances
        # label_ins = np.empty((1,len(tab)))
        # for i in tab:
        #     data_ins[i] = dataset[i]
        #     label_ins[i] = labelset[i]
        
        # # if data_ins == []:
        # #     return [data_ins], [label_ins]
        # # else:
        # # return data_ins, label_ins

    def n_classified_correct_with(self, AN, knn):
        """
        Function counting correctly classified instances
        """
        #data of instances in ANN 
        #excepted labels of instances
        data_for_pred, label_expct = self.find_data_and_labels(AN, self.data.data_all_train, self.data.data_label_train) 
       
        pred = []
        if len(data_for_pred) == 0:
            pred = np.array([])
        else:
            for i in data_for_pred:
                pred.append(knn.predict([i])) 

        #pred = knn.predict(data_for_pred)
        #count correctly classified instances in ANN
        n = 0
        for i in range(len(pred)):
            if pred[i] == label_expct[i]:
                n = n+1
        
        return n


    def n_classified_correct_without(self, index, AN, graph):
        """
        Function counting correctly classified instances without 
        """

        NN_of_A = []

        for i in AN:
            neigh = np.where(graph[i] == 1)[0]
            NN_of_A.append(np.delete(neigh, np.argwhere(neigh == index)))  #indexes of neighbours without actual
        pred = []

        n = 0
        for idx, nn in enumerate(NN_of_A):

            id_for_fit = np.append(nn, AN[idx]) #nn.append(AN[idx])
            data_for_pred, label_expct = self.find_data_and_labels(id_for_fit, self.data.data_all_train, self.data.data_label_train)
            data_of_id, label_of_id =  self.find_data_and_labels([AN[idx]], self.data.data_all_train, self.data.data_label_train)
            if len(data_for_pred) == 0:
                pred.append = np.array([])
                n = n+1
            else:
                knn = KNeighborsClassifier(n_neighbors=self.k).fit(data_for_pred, label_expct)
                pred.append(knn.predict(data_of_id))
                if(label_of_id == knn.predict(data_of_id)):
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
        rem = 0

        knn_k = KNeighborsClassifier(n_neighbors= self.k).fit(self.data.data_all_train, self.data.data_label_train)
        graph_k = knn_k.kneighbors_graph().toarray()

        knn_k1 = KNeighborsClassifier(n_neighbors= self.k + 1).fit(self.data.data_all_train, self.data.data_label_train)
        graph_k1 = knn_k1.kneighbors_graph().toarray()

        NN_k = np.empty((n_instances), dtype=object)
        AN_k = np.empty((n_instances), dtype=object)

        for i in range(n_instances):
            NN_k[i] = np.where(graph_k[i] == 1)[0] #indexes of neighbours
            AN_k[i] = np.where(graph_k[:,i] == 1)[0] #indexes of associates
        
        for i, d in np.ndenumerate(self.red_data): #enumerate(self.red_data[:]):
            n_with = self.n_classified_correct_with(AN_k[i[0]], knn_k) #tu powinno być ANNk
            n_without = self.n_classified_correct_without(i[0], AN_k[i[0]], graph_k1) #tu ANk1

            if (n_without >= n_with):
                rem = rem +1
                """
                remove instances TODO how
                """

                self.red_data = np.delete(self.red_data, [i[0]], axis=0)
                self.red_lab = np.delete(self.red_lab, [i[0]], axis = 0)
                AN_k[i[0]] = np.delete(AN_k[i[0]], [i[0]], axis=0)# np.array([]) #delete from associates

        #print(rem)

        #end time measurement
        end = process_time()

        if return_time:
            return end - start



        


