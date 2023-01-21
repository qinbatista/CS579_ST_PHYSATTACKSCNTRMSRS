import numpy as np


class DataManager:
    def __init__(self, path):
        np.show_config()
        self.__data = np.zeros((1000000, 17))
        self.__path = path
        self.__noise_data = np.zeros((1000000, 16))

    def _import_data(self):
        self.__data = np.genfromtxt(self.__path, delimiter=',')
        self.__noise_data = self.__data[:,0:1]
        pass

if __name__ == '__main__':
    # myDataManager = DataManager('timing_noisy.csv')
    myDataManager = DataManager('timing_noisy_test.csv')
    myDataManager._import_data()
