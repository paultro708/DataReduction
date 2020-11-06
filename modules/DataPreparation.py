import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.datasets import fetch_rcv1
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def create_path_csv(folder, name):
        """
        Function creating string with path of dataset in csv format
        :folder: folder name
        :name: name of dataset
        """
        return folder + '\\' + name + '.csv'

#path of folder with datasets in csv format
datasets_folder = 'D:\Studia\inz\Repos\DataReduction\modules\datasets_csv' # "datasets_csv"

#dictionary of names of datasets and paths
dataset_path = {"iris": create_path_csv(datasets_folder, 'iris'),
                "glass": create_path_csv(datasets_folder, 'glass'),
                "letter": create_path_csv(datasets_folder, 'letter'),
                "liver": create_path_csv(datasets_folder, 'liver'),
                "pendigits": create_path_csv(datasets_folder, 'pendigits'),
                "spambase": create_path_csv(datasets_folder, 'spambase'),
                "segment": create_path_csv(datasets_folder, 'segment'),
                "satimage": create_path_csv(datasets_folder, 'satimage'),
                "yeast": create_path_csv(datasets_folder, 'yeast')
                }


class DataPreparation:
    """
    Class using for loading possible datasets and preparing for reducing algorithms and raports


    Atributes:
    :dataset_name: name of dataset
    """
    # #name of dataset
    # dataset_name = 'iris'
    # #original loaded dataset as pandas data frame
    # dataset=1
    
    # #all original data without labels
    # data_all=1
    # #labels for original data
    # data_label=1

    # #training data obtained from original dataset
    # data_all_train=1
    # #testing data obtained from original dataset
    # data_all_test=1
    # #labels of training data obtained from original dataset
    # data_label_train=1
    # #labels of training data obtained from original dataset
    # data_label_test = 1

    # #number of classes in dataset
    # n_classes=1

    # #array of fetures
    # features=[]

    # #name of column with class labels
    # class_col = 'class'

    # #dictionary of labels of classes and their numerical equivalent index
    # class_dict = dict()
    def load_dataset(self):
        """
        Function loading dataset to pandas dataframe
        """
        

        # if path == 'default':
        #     self.dataset = pd.read_csv(path)
        # else:
        if self.dataset_name not in dataset_path:
            raise Exception("Please select right dataset name!")
        else:
            self.dataset = pd.read_csv(dataset_path[self.dataset_name])

    def prepare_dataset(self):
        """
        Function preparing dataset for reduction
        """
        #if self.class_col: #column name
        #create data frame with class labels

        try: 
            self.data_label = self.dataset[self.class_col]
        except KeyError:
            raise Exception("Please select the existing column name as column with labels!")
            return
        #drop column with label 
        self.data_all = self.dataset.drop(columns=self.class_col)
        #normalize data
        self.data_all = normalize(self.data_all)

        #map labels to 0-n indexes
        self.class_dict = dict()
        i=0
        for label in set(self.data_label):
            self.class_dict[label] = i
            i+=1

        # #init array of features
        self.n_features = self.data_all.shape[1]

        #convert to numpy array
        self.data_all = np.array(self.data_all)
        self.data_label = np.array(self.data_label)

        #split data into train and test
        self.data_all_train, self.data_all_test, self.data_label_train, self.data_label_test = train_test_split(self.data_all, self.data_label, test_size=0.3)
        
        #init number of classes
        self.n_classes = len(set(self.data_label))

        

    def __init__(self, name = "iris", class_col = 'class'):
        """
        Initialize dataset

        :name: name of dataset - one of: 
                "iris", 
                "glass", 
                "letter", 
                "liver", 
                "pendigits", 
                "spambase", 
                "segment",
                "satimage" or
                "yeast"
        """
        self.dataset_name = name
        self.class_col = class_col
        self.load_dataset()
        self.prepare_dataset()
        # try:
        #     self.load_dataset()
        #     self.prepare_dataset()
        # except Exception as e:
        #     print(e)

    
    

    



    @property
    def training(self):
        return self.data_all_train, self.data_label_train

    @training.setter
    def training(self, data, label):
        """
        Setter updating training data after reduction. Using for re-reduction
        :data: training dataset
        :label: labels of training dataset
        """
        self.data_all_train = data
        self.data_label_train = label


    @classmethod
    def from_reduced_dataset(cls):
        """
        TODO
        Aletrnative constructor 
        """
        
        pass


# data = DataPreparation("liver")
# data.load_dataset()
# data.prepare_dataset()

# print(data.features)
# print(data.class_dict)
# print(len(data.data_label))
