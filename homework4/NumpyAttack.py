import numpy as np
from multiprocessing import Pool, Manager
from matplotlib import pyplot as plt
import math
import os
from tqdm import tqdm
import hashlib
import numpy as np
from tqdm import tqdm


class DataManager:
    def __init__(self):
        self.__SBOX = np.array([
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
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ], dtype=np.uint8)

        # Inverse of AES SBOX
        self.__ISBOX = self.__SBOX.argsort()

        # AES MixCols matrix
        self.__MIXCOLS = np.array([[2, 3, 1, 1],
                                   [1, 2, 3, 1],
                                   [1, 1, 2, 3],
                                   [3, 1, 1, 2]])
        self.__load_text()

    def check_keybytes(k_0: int, k_13: int, k_10: int, k_7: int):
        keybytes = bytes([k_0, k_13, k_10, k_7])
        hasher = hashlib.sha3_256()
        hasher.update(keybytes)
        key_hash = hasher.hexdigest()
        if key_hash == '4409976e63e88e6d0ef93405e6b6d678c2a498d22dcaa72b28c8c9cd6233ec7f':
            print("Congratulations! Correct 4 keybytes found")
            return True
        print("Not quite right")
        return False

    def check_pair(self):
        simple_ctxt1 = [174, 44, 204, 43, 18, 196, 238, 88, 3, 227, 92, 0, 137, 106, 205, 88]
        simple_ftxt1 = [128, 44, 204, 43, 18, 196, 238, 171, 3, 227, 159, 0, 137, 186, 205, 88]

        simple_ctxt2 = [41, 4, 148, 29, 23, 74, 41, 127, 125, 148, 36, 219, 29, 127, 4, 58]
        simple_ftxt2 = [186, 4, 148, 29, 23, 74, 41, 160, 125, 148, 59, 219, 29, 172, 4, 58]

        # Load ctext/ftext pairs in the correct AES column order
        simple_ctxt1 = np.reshape(simple_ctxt1, (4, 4), order='F').astype(np.uint8)
        simple_ftxt1 = np.reshape(simple_ftxt1, (4, 4), order='F').astype(np.uint8)

        simple_ctxt2 = np.reshape(simple_ctxt2, (4, 4), order='F').astype(np.uint8)
        simple_ftxt2 = np.reshape(simple_ftxt2, (4, 4), order='F').astype(np.uint8)

        print("First pair diff:")
        print(simple_ctxt1 ^ simple_ftxt1)

        print("Second pair diff:")
        print(simple_ctxt2 ^ simple_ftxt2)

        row_type = np.dtype((np.uint8, (4, 4)))
        all_ctext = np.fromfile("task2/full_dfa_data/ctext.bin", dtype=row_type).transpose(0, 2, 1)
        all_ptext = np.fromfile("task2/full_dfa_data/ptext.bin", dtype=row_type).transpose(0, 2, 1)
        all_ftext = np.fromfile("task2/full_dfa_data/ftext.bin", dtype=row_type).transpose(0, 2, 1)
        print(all_ctext[0] ^ all_ftext[0])

    def __load_text(self):
        row_type = np.dtype((np.uint8, (4, 4)))
        self.__all_ctext = np.fromfile("task2/full_dfa_data/ctext.bin", dtype=row_type).transpose(0, 2, 1)
        self.__all_ptext = np.fromfile("task2/full_dfa_data/ptext.bin", dtype=row_type).transpose(0, 2, 1)
        self.__all_ftext = np.fromfile("task2/full_dfa_data/ftext.bin", dtype=row_type).transpose(0, 2, 1)
        print(self.__all_ctext[0] ^ self.__all_ftext[0])

    def __galois_mult_2(self, a):
        temp = (a << 1) & 0xff
        if (a & 0x80):
            temp ^= 0x1b
        return temp

    def __galois_mult_3(self, a):
        return self.__galois_mult_2(a) ^ a

    def __shift(mat):
        shifted = np.zeros_like(mat)
        for i in range(4):
            shifted[i] = mat[i, np.arange(i, 4+i) % 4]
        return shifted

    def __unshift(mat):
        pass

    def _work_all_mixcols(self):
        D = []
        mixcol = self.__MIXCOLS[:, 0]
        for x in range(1, 255+1):
            D_element = []
            for j in range(4):
                out = None
                if mixcol[j] == 1:
                    out = x
                if mixcol[j] == 2:
                    out = self.__galois_mult_2(x)
                if mixcol[j] == 3:
                    out = self.__galois_mult_3(x)
                D_element.append(out)
            D.append(D_element)
        print("Length of lookup table:", len(D))

if __name__ == '__main__':
    myDataManager = DataManager()
    myDataManager.check_pair()
    myDataManager._work_all_mixcols()
