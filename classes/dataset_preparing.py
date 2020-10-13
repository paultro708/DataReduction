import pandas as pd


class DatasetPreparing:
    def __init__(self, path):
        self.path = path

        try:
            self.data_pd = pd.read_csv(self.path)
        except:
            print('Could not open file')

    
data = DatasetPreparing("D:\Studia\inz\datasets")
print(data.data_pd)
