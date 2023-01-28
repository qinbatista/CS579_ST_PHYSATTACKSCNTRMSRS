import numpy as np
from multiprocessing import Pool, Manager
from TimerTool import TimerTool
from matplotlib import pyplot as plt


class DataManager:
    def __init__(self, path):
        self._timer = TimerTool()
        self.__data = np.memmap(path, dtype='uint8', mode='r')
        self.__n = self.__data.shape[0]
        pass

    def _naive_approach_mean(self):
        sum = self.__data.sum()
        mean = sum / self.__data.shape[0]
        print(f"[Naive Approach]Mean:		{mean}")
        return mean

    def _naive_approach_variance(self):
        sum = self.__data.sum()
        mean = sum / self.__data.shape[0]
        squared_differences = (self.__data-mean)**2
        variance = squared_differences.sum() / self.__data.shape[0]
        print(f"[Naive Approach]Variance:	{variance}")
        return variance

    def _welford_algorithm_mean(self, mean, data):
        self.__n += 1
        mean += (data - mean) / self.__n
        print(f"[Welford Algorithm]Mean:	{mean}")
        return mean

    def _welford_algorithm_variance(self, mean, variance, data):
        self.__n += 1
        delta = data - mean
        mean += delta / self.__n
        variance += (delta*(data - mean)) / (self.__n-1)
        print(f"[Welford Algorithm]Variance:	{variance}")
        return variance


if __name__ == '__main__':
    myDataManager = DataManager('measurement_data_2023_uint8.bin')
    mean = myDataManager._naive_approach_mean()
    variance = myDataManager._naive_approach_variance()
    new_mean = myDataManager._welford_algorithm_mean(mean, 122)
    new_variance = myDataManager._welford_algorithm_variance(new_mean, variance, 122)
