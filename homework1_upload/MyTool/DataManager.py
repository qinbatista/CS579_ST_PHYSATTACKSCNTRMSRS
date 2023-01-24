import numpy as np
from multiprocessing import Pool
from TimerTool import TimerTool


class DataManager:
    def __init__(self, path):
        self._timer = TimerTool()
        self.__data = np.zeros((1000000, 17))
        self.__path = path
        self.__time_data = np.zeros((1000000, 1))
        self.__256bitKey = np.arange(256).reshape(1, 256)
        self.__sbox_table = [
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
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
        self.__vector_look_up_table = np.vectorize(self._look_up_table)
        self.__vector_find_time = np.vectorize(self._find_the_time)
        self.__group0 = np.zeros((16, 1000000, 256))
        self.__group1 = np.zeros((16, 1000000, 256))
        self.__group0Count = np.zeros((16, 1, 256)).astype(np.int32)
        self.__group1Count = np.zeros((16, 1, 256)).astype(np.int32)
        self._import_data()
        self._key = [0]*16

    def _import_data(self):
        self.__data = np.genfromtxt(self.__path, delimiter=',').astype(np.int32)[:, 0:17]
        self.__time_data = self.__data[:, 16:17]
        self.__data = self.__data[:, 0:16]

    def _compute_all_key(self, column_index):
        # generate 0 raw's plain text XOR key
        # first 100000 value
        value_from_plain_and_key = self.__data[:, column_index:column_index+1] ^ self.__256bitKey

        # generate look up table vales
        _MSB_matrix = self.__vector_look_up_table(value_from_plain_and_key) & 0x80
        count_0_group = np.where(_MSB_matrix == 0)
        count_1_group = np.where(_MSB_matrix == 0x80)

        self.__vector_find_time(column_index, count_0_group[0], count_0_group[1], 0)
        self.__vector_find_time(column_index, count_1_group[0], count_1_group[1], 0x80)

        sumGroup0 = np.sum(self.__group0[column_index], axis=0)
        sumGroup1 = np.sum(self.__group1[column_index], axis=0)
        aveSumGroup0 = sumGroup0/self.__group0Count[column_index]
        aveSumGroup1 = sumGroup1/self.__group1Count[column_index]
        value = aveSumGroup1-aveSumGroup0
        self._key[column_index] = np.argmax(value)
        pass

    def _look_up_table(self, value):
        return self.__sbox_table[value]

    def _find_the_time(self, column_index, raw, column, group_index):
        if (group_index == 0):
            if (self.__group1[column_index][raw][column] == 0):
                self.__group0[column_index][raw][column] = self.__group0[column_index][raw][column] + self.__time_data[raw][0]
                self.__group0Count[column_index][0][column] = self.__group0Count[column_index][0][column] + 1
            pass
        if (group_index == 0x80):
            if (self.__group1[column_index][raw][column] == 0):
                self.__group1[column_index][raw][column] = self.__group1[column_index][raw][column] + self.__time_data[raw][0]
                self.__group1Count[column_index][0][column] = self.__group1Count[column_index][0][column] + 1
            pass
       # return self.__time_data[index][0]

    def _print(self):
        print("1")

    def _findKey(self):
        with Pool() as p:
            p.map(self._print, [blur for blur in range(0, 300, 2)])


if __name__ == '__main__':
    # myDataManager = DataManager('timing_noisy.csv')
    myDataManager = DataManager('timing_noisy_test.csv')
    myDataManager._timer._timerStart()
    for i in range(0, 1):
        myDataManager._compute_all_key(i)
    # myDataManager._compute_all_key(2)
    # myDataManager._compute_all_key(3)
    # print(myDataManager._key)
    # myDataManager._timer._timerStop()

    # with Pool(4) as p:
    #     p.map(myDataManager._print, [0, 1, 2, 3])
    myDataManager._timer._timerStop()
    myDataManager._timer._displayExecutionTime()
    print(myDataManager._key)