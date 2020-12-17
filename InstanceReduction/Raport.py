# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
from .DataPreparation import DataPreparation
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
import os
from sklearn.preprocessing import normalize
# from time import process_time
import time
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

classifiers = {'knn': KNeighborsClassifier(), 
                'svm': svm.SVC(), 
                'naive_bayers': GaussianNB(), 
                'decision_tree': DecisionTreeClassifier(),
                'neural_network': MLPClassifier()}

# classify_metrics = {"accuracy": accuracy_score,
#                     "f1": f1_score,
#                     "precision": precision_score,
#                     "recall": recall_score}

class Raport():
    """Class responsible for creating summary of the classification results for the original and reduced data
    It uses DataPreparation instance and arrays of reduced data and labels for reduced data - _Reduction attributes.

    Attributes:
        original (DataPreparation): instance representing original data
        reduced_data (np.ndarray): array with reduced data of training dataset
        reduced_label (np.ndarray): array with labels for the reduced data of training dataset
    """

    def __init__(self, original :DataPreparation, reduced_data: np.ndarray, reduced_label: np.ndarray):
        """Constructor of Raport

        Args:
            original (DataPreparation): instance representing original data
            reduced_data (np.ndarray): array with reduced data of training dataset
            reduced_label (np.ndarray): array with labels for the reduced data of training dataset

        Raises:
            Exception: when reduced_data or reduced_label is empty
        """
        self.original = original
        self.reduced_data = reduced_data
        self.reduced_label = reduced_label
        if 0 in [len(reduced_data), len(reduced_label)]:
            raise Exception("Cannot create raport for empty reduced data")


    def draw_plots(self, colx: str, coly: str, path = None, show:bool = True, save:bool = False):
        """Function creating scatter plots with reduced and original data for given feature names

        Args:
            colx (str): name of column from dataset
            coly (str): name of column from dataset
            path: path where plots will be saved if parameter :save is True. Defaults to None. If :save is True and path has not been given, plots will save in dir 'plots' in working directory 
            show (bool, optional): parameter for showing windows with plots. Defaults to True.
            save (bool, optional): parameter for saving plots in :path. Defaults to False.
        """
        #prepare labels
        orig = []
        for i in self.original.data_label_train:
            orig.append(self.original.class_dict[i])
        orig = np.array(orig)
        red = []
        for i in self.reduced_label: 
            red.append(self.original.class_dict[i])
        red = np.array(red)

        #get indexes of features
        idx = self.original.features.index(colx)
        idy = self.original.features.index(coly)


        #create plots
        for name, obj, col in [('Original dataset', self.original.data_all_train, orig), ('Reduced dataset', self.reduced_data, red)]:
            fig, ax = plt.subplots()
            scatter = ax.scatter(obj[:,idx], obj[:,idy], c = col)
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
            
            plt.xlabel(self.original.features[idx])
            plt.ylabel(self.original.features[idy])
            #set same scale
            if name == 'Original dataset':
                bottom_y, top_y = plt.ylim()
                bottom_x, top_x = plt.xlim()
            else:
                plt.xlim(bottom_x, top_x)
                plt.ylim(bottom_y, top_y)
            plt.title(name)
            if save:
                if path == None:
                    path = os.path.join(os.getcwd(), 'plots')
                if not os.path.exists(path):
                    os.makedirs(path)
                filename = "{} {}({})".format(name, self.original.features[idx], self.original.features[idy])        
                plt.savefig(os.path.join(path, filename))
            if show:
                plt.show()


    @staticmethod
    def create_confusion_matrix(classifier, test_set, test_labels, title:str, filename:str, path, show:bool, save:bool):
        """Function creating confusion matrix for given parametres

        Args:
            classifier: sklearn classifier, given from the available ones
            test_set: test dataset
            test_labels: labels for test dataset
            title (str): title of plot
            filename (str): name of the file that will be saved if :save is True
            path: path where plot
            show (bool): parameter for showing window with created plot
            save (bool): parameter for saving plot as file with name :filename in :path
        """
        plot = plot_confusion_matrix(classifier, test_set, test_labels, cmap=plt.cm.Blues)
        #title = "Confusion matrix\n reduced data - classifier: " + str(c_type)
        plot.ax_.set_title(title)
        if save:
            if path == None:
                path = os.path.join(os.getcwd(), 'plots')
            if not os.path.exists(path):
                os.makedirs(path)
            # else if not os.path.isabs(filepath):
            #     path = os.path.join(os.getcwd(), path)        
            plot.figure_.savefig(os.path.join(path,filename))
        if show:
            plt.show()

    def _raport(self, original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_type, show_cf_matrix :bool, path, save_cf_matrix: bool, norm: bool):
        """Main function in Raport class. It is responsible for :
        - classify with original and reduced dataset,  
        - printing results of classification quality
        - creating confusion matrices
        - time measurement for classification and prediction

        Args:
            original_set: array with original dataset
            original_labels: array with labels for original datase
            reduced_set: array with reduced dataset
            reduced_labels: array with labels for reduced dataset
            test_set: array with test dataset
            test_labels: array with labels for test dataset
            c_type (str): type of classifier. If 'all' - creates raport for all available classifiers
            show_cf_matrix (bool): parameter for showing windows with created confusion matrices
            path: path to save in created confusion matrices
            save_cf_matrix (bool): parameter for saving created confusion matrices

        Raises:
            Exception: when given value classifier type not exist in dictionary of available types
        """
        if (c_type !='all') and (c_type not in classifiers):
            raise Exception("Classifier type not exist in available set!")
        else:
            if c_type == 'all':
                for c_t in classifiers:
                    self._raport(original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_t, show_cf_matrix, path, save_cf_matrix, norm)

            else:
                #select classifier
                classifier = classifiers[c_type]

                #normalize data:
                if norm:
                    original_set = normalize(original_set)
                    reduced_set = normalize(reduced_set)
                
                #train with original dataset and time measure
                start = time.clock()
                classifier.fit(original_set, original_labels)
                end = time.clock()
                training_time = end - start

                #make predictions and time measure
                start = time.clock()
                predict = classifier.predict(test_set)
                end = time.clock()
                prediction_time = end - start

                #create confusion matrix
                if (save_cf_matrix or show_cf_matrix) is not False:
                    title = "Confusion matrix\n original data - classifier: {}".format(str(c_type))
                    self.create_confusion_matrix(classifier, test_set, test_labels, title, "Original - " + c_type, path, show_cf_matrix, save_cf_matrix)

                #print raport with metrics for original training data
                print('=============')
                print("Classifier:  ", c_type)
                print('=============')
                print("Raport for original dataset")
                print('Count of instances: ', len(original_labels))
                print(classification_report(test_labels, predict, digits=4)) 
                print("Cohen's Kappa: {:.2f}".format(cohen_kappa_score(test_labels, predict)))
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time)

                #same for reduced training dataset
                classifier = classifiers[c_type]
                #train
                start = time.clock()
                classifier.fit(reduced_set, reduced_labels)
                end = time.clock()
                training_time = end - start
                #predict
                start = time.clock()
                predict = classifier.predict(test_set)
                end = time.clock()
                prediction_time = end - start

                #create confusion matrix
                if (save_cf_matrix or show_cf_matrix) is not False:
                    title = "Confusion matrix\n reduced data - classifier: {}".format(str(c_type))
                    self.create_confusion_matrix(classifier, test_set, test_labels, title, "Reduced - " + c_type, path, show_cf_matrix, save_cf_matrix)

                print("\nRaport for reduced dataset")
                print('Count of instances: ', len(reduced_labels))
                print(classification_report(test_labels, predict, digits = 4))
                print("Cohen's Kappa: {:.2f}".format(cohen_kappa_score(test_labels, predict)))
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time, "\n")
                print('Reduction factor: {:.2f} %'.format((len(original_labels) - len(reduced_labels))/len(original_labels)*100))
                print('===')
                row = pd.Series([c_type, 
                                    accuracy_score(test_labels, predict),
                                    cohen_kappa_score(test_labels, predict),
                                    training_time, prediction_time
                                    ])
        


    def print_raport(self, c_type= 'all', show_cf_matrix = True, path = None, save_cf_matrix = False, norm = False):
        
        """Function responssible for call function printing raport with apriopriate arguments. 

        Args:
            c_type (str, optional): Type of classifier. One from dictionary :classifiers:. Defaults to 'all'. 
                                    It means that raport will be created for all available classifiers.
                                    Possible values: 'knn': KNeighborsClassifier(), 
                                            'svm': svm.SVC(), 
                                            'naive_bayers': GaussianNB(), 
                                            'decision_tree': DecisionTreeClassifier(),
                                            'neutral_network': MLPClassifier().
            show_cf_matrix (bool, optional): Parameter for showing windows with confusion matrices. Defaults to True.
            path (optional): Path for saving created confusion matrices. Defaults to None.
            save_cf_matrix (bool, optional): Parameter for saving created confusion matrices. Defaults to False.
        """
        self._raport(self.original.data_all_train, 
                    self.original.data_label_train, 
                    self.reduced_data, 
                    self.reduced_label, 
                    self.original.data_all_test, 
                    self.original.data_label_test, 
                    c_type,
                    show_cf_matrix,
                    path,
                    save_cf_matrix,
                    norm)