from MyTool.TimerTool import HackTool
from MyTool.DataManager import DataManager
from multiprocessing import Pool
import numpy as np
from MyTool.AESEncrypt import AESEncrypt


def Question1():
    # 1: Please consider the following AES inputs and compute the output after the first SubBytes operation.
    # Only use a pocket calculator and the table given in Figure
    # 1. This is intended to make sure you fully understood the concept. (10 pts)
    myAE = AESEncrypt()
    # 1:generate 4*4 block
    _16bit_block_text = myAE._import_plain_text((0x00, 0x00, 0x00, 0x00,
                                                 0x00, 0x00, 0xc1, 0xa5,
                                                 0x51, 0xf1, 0xed, 0xc0,
                                                 0xff, 0xee, 0xb4, 0xbe))
    around_key = np.array([[0x00, 0x00, 0x01, 0x02],
                           [0x03, 0x04, 0xde, 0xca],
                           [0xf0, 0xc0, 0xff, 0xee],
                           [0x00, 0x00, 0x00, 0x00],
                           ], dtype=np.uint8)
    # 2:generate sub bytes from Sbox
    _sub_bytes = myAE._convert_to_sub_byte(_16bit_block_text)
    _my_string = "1️⃣ convert to subByte:	"
    for i in range(4):
        for j in range(4):
            _my_string += hex(_sub_bytes[i][j])+" "
    print(_my_string)
    # 3:shift row
    _shifted_raw = myAE._shit_row(_sub_bytes)
    _my_string = "2️⃣ shifted row:		"
    for i in range(4):
        for j in range(4):
            _my_string += hex(_shifted_raw[i][j])+" "
    print(_my_string)
    # 4:mix column
    _mixed_column = myAE._mix_column(_shifted_raw)
    _my_string = "3️⃣ mixed column:		"
    for i in range(4):
        for j in range(4):
            _my_string += hex(_shifted_raw[i][j])+" "
    print(_my_string)
    # 5:add round key, first around
    _AddRoundKey = myAE._add_round_key(_mixed_column, around_key)
    _my_string = "4️⃣ added round key:	"
    for i in range(4):
        for j in range(4):
            _my_string += hex(_AddRoundKey[i][j])+" "
    print(_my_string)


if __name__ == '__main__':
    # initialize all the tools
    Question1()
    myDataManager = DataManager('timing_noisy.csv')
    myHackTool = HackTool()
    # myHackTool._maxCoreProcessing(myFunction, [givingValue for givingValue in range(0, 1)])
    # start the processing
