import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
import os
from pandas.api.types import is_numeric_dtype

class NotLoadedException(Exception):
    """Class representing exception raising after trying to prepare the dataset when the dataset has not yet been loaded

    Args:
        Exception: exception python class
    """
    def __init__(self, message: str):
        """Constructor of NotLoadedException

        Args:
            message (str): the message shown when the exception occured
        """
        self.message = message

#path of dir with datasets in csv format
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_csv")
extension ='.csv'

dataset_path = {"iris": os.path.join(datasets_dir, 'iris'+extension),
                "glass": os.path.join(datasets_dir, 'glass' + extension),
                "letter": os.path.join(datasets_dir, 'letter' + extension),
                "liver": os.path.join(datasets_dir, 'liver'+ extension),
                "pendigits": os.path.join(datasets_dir, 'pendigits'+ extension),
                "spambase": os.path.join(datasets_dir, 'spambase'+ extension),
                "segment": os.path.join(datasets_dir, 'segment'+ extension),
                "satimage": os.path.join(datasets_dir, 'satimage'+ extension),
                "shuttle_train": os.path.join(datasets_dir, 'shuttle_training'+ extension),
                "yeast": os.path.join(datasets_dir, 'yeast'+ extension),
                "electricity": os.path.join(datasets_dir, 'electricity'+ extension),
                "fried": os.path.join(datasets_dir, 'fried'+ extension),
                "avila": os.path.join(datasets_dir, 'avila'+ extension)
                }

class DataPreparation:
    """
    Class using for loading possible datasets and preparing for reducing algorithms and raports

    Attributes:
        class_col: column marked as representing as column with labels for decission class. Can be str name of column or integer index.
        dataset: dataset loaded as pandas data frame. It must have only numerical, non missed values.
        data_all: numpy array with data without labels
        data_label: numpy arrray with labels for data
        features: list with feature names - header without :class_col
        n_features: count of features
        n_classes: count of decision classes
        class_dict (dict): dictionary with class names as keys and mapped integer representation as values
        data_all_train: part of :data_all representing train subset. It constitute about 0.7 of :data_all. This subset will be used in the reduction process.
        data_all_test: part of :data_all representing test subset. It constitute about 0.3 of :data_all. This subset will be used for evalueating the classification quality.
        data_label_train: labels of instances in :data_all_train
        datat_label_test: labels of instances in :data_all_test

    """
    @staticmethod
    def normalize(array: np.ndarray):
        """Function normalizing numpy array.

        Args:
            array (np.ndarray): array to normalize

        Returns:
            normalized (np.ndarray): normalized :array
            weights (np.ndarray): weights of normalization
        """
        normalized = np.copy(array).astype('float64')
        weights = np.ones(array.shape[1])
        for i in range(len(weights)):
            weights[i] = np.sqrt(sum(normalized[i]**2))
        for i in range(len(weights)): 
            normalized[:,i] /= weights[i]
        return normalized, weights

    @staticmethod
    def reverse_normalize(array: np.ndarray, weights: np.ndarray):
        """Function reverse normalizing numpy array - back to values before normalization

        Raises:
            Exception: when number of weights and columns does not agree

        Args:
            array (np.ndarray): array to reverse normalizing
            weights (np.ndarray): wights used for normalizing

        Returns:
            (np.ndarray): array with values before normalization
        """
        if array.shape[1] != len(weights):
            raise Exception('Number of weights does not agree with shape[1] of array')
        arr_before = array
        for i in range(len(weights)): 
            arr_before[:,i] *= weights[i]
        return arr_before

    def prepare_dataset(self):
        """
        Function preparing dataset for reduction - getting out metadata and split dataset to train and test subsets.
        """
        #map labels to 0-n indexes
        self.class_dict = dict()
        i=0
        for label in set(self.data_label):
            self.class_dict[label] = i
            i+=1

        #init array of features
        self.n_features = self.data_all.shape[1]

        #convert to numpy array
        self.data_all = np.array(self.data_all)
        self.data_label = np.array(self.data_label)

        #split data into train and test
        self.data_all_train, self.data_all_test, self.data_label_train, self.data_label_test = train_test_split(self.data_all, 
                                                                                                                self.data_label, 
                                                                                                                test_size=0.3,
                                                                                                                random_state = 42)
        
        #init number of classes
        self.n_classes = len(set(self.data_label))


    def load_csv(self):
        """Function checking whether the loaded data set meets the basic requirements and getting out metadata

        Raises:
            NotLoadedException: when dataset has not been loaded
            Exception: when dataset is empty, has not enough count of columns, contains missing values or contains nonumeric values
            IndexError: when the given index of column selected as class column is out of range
            TypeError: when the given column selected as class column is wrong type. 
            :class_col must be str type with the name of column or int means index of column
            KeyError: when the given column name not exists in dataset 
        """
        #check loaded
        try:
            self.dataset == None
        except NameError:
            raise NotLoadedException('Dataset must be loaded before preparation')

        #is empty
        if self.dataset.empty:
            raise Exception('Dataset is empty!')

        #constains more than one column
        if len(self.dataset.columns) < 2:
            raise Exception('Dataset must have minimum 2 columns! Please check if you select apriopriate filepath or separator.')

        #contains missing values
        if self.dataset.isnull().values.any() == True:
            raise Exception('Dataset contains Nan values. Please fill missing values before use class.')

        #constains class column
        #check if it is index
        if self.class_col not in self.dataset.columns:
            if type(self.class_col) == int:
                if self.class_col > len(self.dataset.columns):
                    raise IndexError('Index of class column is out of range')
                self.data_label = self.dataset.iloc[:,self.class_col]
                self.data_all = self.dataset.drop(self.dataset.columns[self.class_col], asix=1)
            elif type(self.class_col) != str:
                raise TypeError('Class column must be type str or int')
            else:
                raise KeyError('Selected class column {} is wrong. Please select existing column name or index'.format(self.class_col))
        else:
            self.data_label = self.dataset[self.class_col]
            self.data_all = self.dataset.drop(columns=self.class_col)

        self.features = self.data_all.columns.values.tolist()

        #check if constains only numeric values
        for col in self.data_all.columns:
            if not is_numeric_dtype(self.data_all[col]):
                raise Exception('Dataset constains non numeric values.')
    

    def __init__(self, name:str = None, filepath = None, class_col = 'class', sep = ','):
        
        """Constructor of DataPreparation. It tries open file with pandas library. 
        After that, loaded dataset is prepared for subsequent actions by calling apriopriate function.

        Args:
            name (str, optional): name of dataset, one of available:
                            "iris", 
                            "glass", 
                            "letter", 
                            "liver", 
                            "pendigits", 
                            "spambase", 
                            "segment",
                            "satimage" or
                            "yeast".
                    Default to None. If not given - :filepath should be.
            filepath (optional): path of file containing selected dataset. Even if given, when :name also is, named dataset will be loaded. Default to None.
            class_col (oprional): name of column with labels marking decision class. Default to 'class'
            sep (optional): separator used in dataset file for separating values

        Raises:
            TypeError: when type of the given parameter is not apriopriate
            ValueError: when then value of the given parameter is not apriopriate
            FileNotFoundError: when the given file could not be found
            OSError: when the given file could not be opened
            Exception: when the given parameters or file are not appropriate for mapping data
        """
        
        self.class_col = class_col

        #if selected availiable dataset by name
        if name is not None:
            if type(name) != str:
                raise TypeError('Wrong type of dataset name. Please select apriopriate name with str type')
            if name not in dataset_path:      
                raise ValueError('Dataset {} not found. Please select apriopriate dataset name!'.format(name))
            else:
                self.dataset = pd.read_csv(dataset_path[name])
                self.load_csv()
                self.prepare_dataset()

        #if selected file
        elif filepath is not None:
            if not os.path.isabs(filepath):
                filepath = os.path.join(os.getcwd(), filepath)
            if type(sep) != str:
                raise TypeError('Can not use {} separator. Separator must be str type.'.format(sep))
            try: 
                self.dataset = pd.read_csv(filepath, sep = sep)
                self.load_csv()
                self.prepare_dataset()
            except FileNotFoundError:
                raise FileNotFoundError("File {} not found. Please select the existing csv file!".format(filepath))
            except OSError:
                raise OSError('Can not load file from path {}. Please select the apriopriate filepath!'.format(filepath))
            except ValueError:
                raise ValueError('Can not use file in path {}. Please select filepath with apriopriate extension!'.format(filepath))
            # except:
            #     raise Exception('Please select csv filepath with apriopriate extension.')
        
        else:
            raise Exception('Can not load data without name or filepath of dataset file!')



if __name__ == "__main__":
    #Test DataPreparation
    data = DataPreparation()
    print(data)
