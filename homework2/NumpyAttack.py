import numpy as np
from multiprocessing import Pool, Manager
from TimerTool import TimerTool
from matplotlib import pyplot as plt


class DataManager:
    def __init__(self, path):
        self._timer = TimerTool()
        self._data = np.memmap(path, dtype='uint8', mode='r')
        self.__n = self._data.shape[0]
        pass

    def _naive_approach_mean(self):
        sum = self._data.sum()
        mean = sum / self._data.shape[0]
        print(f"[Naive Approach]Mean:		{mean}")
        return mean

    def _naive_approach_variance(self):
        sum = self._data.sum()
        mean = sum / self._data.shape[0]
        squared_differences = (self._data-mean)**2
        variance = squared_differences.sum() / self._data.shape[0]
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

    def _one_pass(self, data_list):
        self._timer._start()
        mean = 0
        variance = 0
        mean_variance = 0
        M2 = 0
        n = 0
        for data in data_list:
            n += 1
            mean += (data - mean) / n

            delta = (data - mean_variance)
            mean_variance += delta / n
            M2 += delta * (data - mean_variance)
        variance = M2 / n
        self._timer._stop()
        self._timer._display()
        print(f"[One Pass]Mean:			{mean}")
        print(f"[One Pass]Variance:		{variance}")

    def _histogram_method(self, data):
        self._timer._start()
        hist, bin_edges = np.histogram(data, bins=3)
        # Calculate the mean
        mean = np.sum(hist * bin_edges[:-1]) / data.shape[0]
        variance = np.sum((bin_edges[:-1]-mean)**2)/data.shape[0]
        self._timer._stop()
        self._timer._display()
        print(f"[Histogram Method]Histogram Mean:	{mean}")
        print(f"[Histogram Method]Histogram Variance:	{variance}")


if __name__ == '__main__':
    myDataManager = DataManager('measurement_data_2023_uint8.bin')
    # print("----------naive approach-----------")
    # mean = myDataManager._naive_approach_mean()
    # variance = myDataManager._naive_approach_variance()
    # print("----------welford algorithm-----------")
    # new_mean = myDataManager._welford_algorithm_mean(mean, 122)
    # new_variance = myDataManager._welford_algorithm_variance(new_mean, variance, 122)
    a = myDataManager._data[:100000]
    print("----------One Pass-----------")
    myDataManager._one_pass(a)
    print("----------Histogram Method-----------")
    myDataManager._histogram_method(a)
