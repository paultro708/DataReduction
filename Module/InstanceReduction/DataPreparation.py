import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
from pandas.api.types import is_numeric_dtype

class NotLoadedException(Exception):
    def __init__(self, message):
        self.message = message

def create_path_csv(folder, name):
        """
        Function creating string with path of dataset in csv format
        :folder: folder name
        :name: name of dataset
        """
        return folder + '\\' + name + '.csv'

#path of folder with datasets in csv format
datasets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_csv") # "datasets_csv"

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
    
    # def load_dataset(self):
    #     """
    #     Function loading dataset to pandas dataframe
    #     """
        

    #     # if path == 'default':
    #     #     self.dataset = pd.read_csv(path)
    #     # else:
    #     if self.dataset_name not in dataset_path:
    #         raise Exception("Please select right dataset name!")
    #     else:
    #         self.dataset = pd.read_csv(dataset_path[self.dataset_name])

    def prepare_dataset(self):
        """
        Function preparing dataset for reduction
        """
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


    def load_csv(self):
    
        #check loaded
        try:
            self.dataset == None
        except NameError:
            raise NotLoadedException('Dataset must be loaded before preparation')

        #constains more than one column
        if len(self.dataset.columns) < 2:
            raise Exception('Dataset must have minimum 2 columns! Please check if you select apriopriate filepath or separator.')

        #contains missing values
        if self.dataset.isnull().values.any() == True:
            raise Exception('Dataset contains Nan values. Please fill missing values before use class.')

        #constains class column
        if self.class_col in self.dataset.columns:
            self.data_label = self.dataset[self.class_col]
            self.data_all = self.dataset.drop(columns=self.class_col)
            self.features = self.data_all.columns.values.tolist()
        else:
            raise Exception('Please select existing column name as class column.')

        #constains only numeric values
        for col in self.data_all.columns:
            if not is_numeric_dtype(self.data_all[col]):
                raise Exception('Dataset constains non numeric values.')
    

    def __init__(self, name = None, filepath = None, class_col = 'class', sep = ','):
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
        if (name is None) and (filepath is None):
            raise Exception('Can not load data without name of dataset or filepath of dataset file!')
        
        self.class_col = class_col

        #if selected file
        if filepath:
            try: 
                self.dataset = pd.read_csv(filepath, sep = sep)
                self.load_csv()
                self.prepare_dataset()
            except FileNotFoundError:
                raise Exception("File {} not found. Please select the existing csv file!".format(filepath))
            except OSError:
                raise Exception('Can not use file in path {}. Please select the apriopriate filepath!'.format(filepath))
            except ValueError:
                raise Exception('Can not use file in path {}. Please select filepath with apriopriate extension!'.format(filepath))
            except:
                raise Exception('Please select csv filepath with apriopriate extension.')
        
        #if selected availiable dataset by name
        elif name:
            if name not in dataset_path:
                raise Exception('Dataset {} not found. Please select apriopriate dataset name!'.format(name))
            else:
                self.dataset = pd.read_csv(dataset_path[name])
                self.load_csv()
                self.prepare_dataset()



if __name__ == "__main__":
    #Test DataPreparation
    data = DataPreparation()
    print(data)
