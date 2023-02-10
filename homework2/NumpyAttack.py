import numpy as np
from multiprocessing import Pool, Manager
from TimerTool import TimerTool
from matplotlib import pyplot as plt
import math


class DataManager:
    def __init__(self, measurement_data_2023_uint8_path, traces_10000x50_int8_path, plaintext_10000x16_uint8):
        self._timer = TimerTool()
        self._data_measurement = np.memmap(measurement_data_2023_uint8_path, dtype='uint8', mode='r')
        self._data_measurement = self._data_measurement[:]  # Test part
        self.__n_measurement = self._data_measurement.shape[0]
        self.__256bitKey = np.arange(256).reshape(1, 256)
        self._sbox_table = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16])

        self._data_trace = np.memmap(traces_10000x50_int8_path, dtype='int8', mode='r').reshape(10000, 50)
        self.__n_trace = self._data_trace.shape

        self._data_plaintext = np.memmap(plaintext_10000x16_uint8, dtype='uint8', mode='r').reshape(10000, 16)
        self.__n_plaintext = self._data_plaintext.shape
        pass

    def _naive_approach_mean(self):
        self._timer._start()
        squared_differences = 0
        sum = 0
        for i in range(self._data_measurement.shape[0]):
            sum += self._data_measurement[i]
        mean = sum / self._data_measurement.shape[0]
        print(f"[Naive Approach]Mean:		{mean}, correct value:{self._data_measurement.mean()}")

        for i in range(self._data_measurement.shape[0]):
            squared_differences = squared_differences + (self._data_measurement[i]-mean)**2
        variance = squared_differences / self._data_measurement.shape[0]
        print(f"[Naive Approach]Variance:	{variance}, correct value:{self._data_measurement.var()}")
        self._timer._stop()
        self._timer._display()
        return mean

    def _welford_algorithm(self):
        self._timer._start()
        aggregate = (0, 0, 0)
        for x in range(self._data_measurement.shape[0]):
            aggregate = self.update(aggregate, self._data_measurement[x])
        mean, variance, sampleVariance = self.finalize(aggregate)
        print(f"[Welford Approach]Variance:	{variance}, correct value:{self._data_measurement.var()}")
        print(f"[Welford Approach]Mean:		{mean}, correct value:{self._data_measurement.mean()}")
        self._timer._stop()
        self._timer._display()

    def update(self, existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)

    # Retrieve the mean, variance and sample variance from an aggregate
    def finalize(self, existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)

    def _one_pass(self):
        self._timer._start()
        mean = 0
        variance = 0
        mean_variance = 0
        M2 = 0
        n = 0
        for i in range(self._data_measurement.shape[0]):
            n += 1
            mean += (self._data_measurement[i] - mean) / n

            delta = (self._data_measurement[i] - mean_variance)
            mean_variance += delta / n
            M2 += delta * (self._data_measurement[i] - mean_variance)
        variance = M2 / (n-1)
        print(f"[One Pass]Mean:		{mean}, 	correct value:{self._data_measurement.mean()}")
        print(f"[One Pass]Variance:	{variance},	correct value:{self._data_measurement.var()}")
        self._timer._stop()
        self._timer._display()

    def _histogram_method(self):
        self._timer._start()
        histogram = np.zeros((256))
        for i in range(self._data_measurement.shape[0]):
            histogram[self._data_measurement[i]] += 1
        sum = 0
        for i in range(histogram.shape[0]):
            sum = sum + histogram[i] * i
        mean = sum / self._data_measurement.shape[0]

        squared_differences = 0
        for i in range(histogram.shape[0]):
            if histogram[i] != 0.0:
                squared_differences = squared_differences + ((i-mean)**2)*histogram[i]
        variance = squared_differences / self._data_measurement.shape[0]
        print(f"[Histogram Method]Histogram Mean:		{mean},	 correct value:{self._data_measurement.mean()}")
        print(f"[Histogram Method]Histogram Variance:	{variance},	 correct value:{self._data_measurement.var()}")
        self._timer._stop()
        self._timer._display()

    def _signal(self):
        signal = np.zeros((50))
        for trace_index in range(0, 50):
            mean_256 = np.zeros((256))
            for p_value in range(0, 256):  # each column of plain text with n go through all trace
                plainText = np.where(self._data_plaintext[:, 0:1] == p_value, 1, 0)  # extract all plain text equal n, n is 1,2,3,...,255
                all_position = np.where(plainText * self._data_trace[:, trace_index:trace_index+1]!=0)
                all_value = np.take(self._data_trace[:, trace_index:trace_index+1], all_position[0])
                the_mean = all_value.mean()
                if np.isnan(the_mean):
                    continue
                mean_256[p_value] = all_value.mean()
            non_zero = mean_256!=0.0
            mean_non_zero = mean_256[non_zero]
            signal[trace_index] = np.var(mean_non_zero)
        fig, ax = plt.subplots()
        ax.plot(signal)
        return signal

    def _noise(self):
        noise = np.zeros((50))
        for trace_index in range(0, 50):
            var_256 = np.zeros((256))
            for p_value in range(0, 256):  # each column of plain text with n go through all trace
                plainText = np.where(self._data_plaintext[:, 0:1] == p_value, 1, 0)  # extract all plain text equal n, n is 1,2,3,...,255
                all_position = np.where(plainText * self._data_trace[:, trace_index:trace_index+1]!=0)
                all_value = np.take(self._data_trace[:, trace_index:trace_index+1], all_position[0])
                the_mean = all_value.var()
                if np.isnan(the_mean):
                    continue
                var_256[p_value] = the_mean
            non_zero = var_256!=0.0
            mean_non_zero = var_256[non_zero]
            noise[trace_index] = np.mean(mean_non_zero)
        fig, ax = plt.subplots()
        ax.plot(noise)
        return noise

    def _SNR(self):
        value = self._signal()/self._noise()
        fig, ax = plt.subplots()
        ax.plot(value)

    def _CPA(self):
        key = []
        correlation = []
        for column in range(0, 16):
            candidate_keys_values = self._data_plaintext[:, column:column+1] ^ self.__256bitKey
            SBOX_matrix = np.take(self._sbox_table, candidate_keys_values)

            # power Model

            trace_column_mean = np.mean(self._data_trace, axis=0).reshape(1, self.__n_trace[1])
            t_dj = self._data_trace-trace_column_mean  # t_dj- mean_t
            keys_column_mean = np.mean(SBOX_matrix, axis=0)
            h_dj = SBOX_matrix-keys_column_mean  # h_dj- mean_h

            r_ij = np.zeros((50, 256))
            for trace_id in range(0, 50):
                # for key_index in range(0, 255):
                up_value = np.sum(h_dj[:, :]*t_dj[:, trace_id:trace_id+1], axis=0)
                value1 = np.sum(h_dj[:, :]**2, axis=0)
                value2 = np.sum(t_dj[:, trace_id:trace_id+1]**2)
                value3 = np.sqrt(value1*value2)
                value = up_value / value3
                r_ij[trace_id] = value
            # fig, ax = plt.subplots()
            # ax.plot(r_ij)
            # pass
            correlation.append(np.amax(r_ij))
            flat_index = np.argmax(r_ij, axis=None)
            index = np.unravel_index(flat_index, r_ij.shape)[1]
            key.append(index)
        correlation = np.around(correlation, decimals=4)
        print(f"Key: {key}")
        string_append = ""
        for i in range(0, 16):
            string_append+=f"{correlation[i]} "
        print(f"Correlation: {string_append}")


if __name__ == '__main__':
    myDataManager = DataManager('measurement_data_2023_uint8.bin', 'traces_10000x50_int8.bin', 'plaintext_10000x16_uint8.bin')
    # mean = myDataManager._naive_approach_mean()
    # new_mean = myDataManager._welford_algorithm()
    # myDataManager._one_pass()^
    # myDataManager._histogram_method()
    # myDataManager._signal()
    # myDataManager._noise()
    # myDataManager._SNR()
    myDataManager._CPA()
    pass
