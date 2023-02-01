import numpy as np
from multiprocessing import Pool, Manager
from TimerTool import TimerTool
from matplotlib import pyplot as plt


class DataManager:
    def __init__(self, measurement_data_2023_uint8_path, traces_10000x50_int8_path, plaintext_10000x16_uint8):
        self._timer = TimerTool()
        self._data_measurement = np.memmap(measurement_data_2023_uint8_path, dtype='uint8', mode='r')
        self.__n_measurement = self._data_measurement.shape[0]

        self._data_trace = np.memmap(traces_10000x50_int8_path, dtype='int8', mode='r').reshape(10000, 50)
        self.__n_trace = self._data_trace.shape

        self._data_plaintext = np.memmap(plaintext_10000x16_uint8, dtype='uint8', mode='r').reshape(10000, 16)
        self.__n_plaintext = self._data_plaintext.shape
        pass

    def _naive_approach_mean(self):
        sum = self._data_measurement.sum()
        mean = sum / self._data_measurement.shape[0]
        print(f"[Naive Approach]Mean:		{mean}, correct value:{self._data_measurement.mean()}")
        return mean

    def _naive_approach_variance(self):
        sum = self._data_measurement.sum()
        mean = sum / self._data_measurement.shape[0]
        squared_differences = (self._data_measurement-mean)**2
        variance = squared_differences.sum() / self._data_measurement.shape[0]
        print(f"[Naive Approach]Variance:	{variance}, correct value:{self._data_measurement.var()}")
        return variance

    def _welford_algorithm_mean(self, mean, data):
        self.__n_measurement += 1
        mean += (data - mean) / self.__n_measurement
        print(f"[Welford Algorithm]Mean:	{mean}, added new data:{data}")
        return mean

    def _welford_algorithm_variance(self, mean, variance, data):
        self.__n_measurement += 1
        delta = data - mean
        mean += delta / self.__n_measurement
        variance += (delta*(data - mean)) / (self.__n_measurement-1)
        print(f"[Welford Algorithm]Variance:	{variance}, added new data:{data}")
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
        variance = M2 / (n-1)
        self._timer._stop()
        self._timer._display()
        print(f"[One Pass]Mean:			{mean}, correct value:{data_list.mean()}")
        print(f"[One Pass]Variance:		{variance},correct value:{data_list.var()}")

    def _histogram_method(self, data):
        self._timer._start()
        hist, bin_edges = np.histogram(data, bins=10)
        mean = np.sum(hist * bin_edges[:-1]) / len(data)
        variance = np.sum((bin_edges[:-1] - mean) ** 2 * hist) / len(data)
        self._timer._stop()
        self._timer._display()
        print(f"[Histogram Method]Histogram Mean:	{mean}, correct value:{data.mean()}")
        print(f"[Histogram Method]Histogram Variance:	{variance}, correct value:{data.var()}")

    def _numerator_of_traces(self):
        mean_trance = np.mean(self._data_trace)
        signal = self._data_trace - mean_trance
        signal = np.sum(signal,axis=0)/self.__n_trace[0]
        plt.plot(signal)
        return signal

    def _denominator_of_traces(self):
        noise = np.std(self._data_trace, axis=0)
        plt.plot(noise)
        return noise

if __name__ == '__main__':
    myDataManager = DataManager('measurement_data_2023_uint8.bin', 'traces_10000x50_int8.bin', 'plaintext_10000x16_uint8.bin')
    # print("----------naive approach-----------")
    mean = myDataManager._naive_approach_mean()
    variance = myDataManager._naive_approach_variance()
    # print("----------welford algorithm-----------")
    # new_mean = myDataManager._welford_algorithm_mean(mean, 122)
    # new_variance = myDataManager._welford_algorithm_variance(variance, variance, 122)
    a = myDataManager._data_measurement[:100000]
    # print("----------One Pass-----------")
    # myDataManager._one_pass(a)
    # print("----------Histogram Method-----------")
    # myDataManager._histogram_method(a)
    # signal = myDataManager._numerator_of_traces()
    # noise = myDataManager._denominator_of_traces()
    # value = signal/noise
    pass
