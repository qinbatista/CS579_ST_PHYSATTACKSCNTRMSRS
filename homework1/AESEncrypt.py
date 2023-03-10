import numpy as np
import galois


class AESEncrypt:
    def __init__(self):
        self.__Sbox = np.zeros((16, 16), dtype=np.uint8)
        self.__sbox_list = (
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
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16)
        self.__Sbox = self.__convert_sbox_to_hex(self.__sbox_list)
        self.__mix_column_matrix = np.array([[0x2, 0x3, 0x1, 0x1],
                                             [0x1, 0x2, 0x3, 0x1],
                                             [0x1, 0x1, 0x2, 0x3],
                                             [0x3, 0x1, 0x1, 0x2]], dtype=np.uint8)
        self.__GF = galois.GF(2**8)

    def __convert_sbox_to_hex(self, sbox_list):
        _Sbox = np.zeros((16, 16), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                _Sbox[i][j] = sbox_list[i * 16 + j]
                # print(f"[{i}:{j}]:{hex(_Sbox[i][j])}")
        return _Sbox

    def __convert_plain_text_to_block(self, painText_list):
        _block = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                _block[i][j] = painText_list[i * 4 + j]
                # print(f"[{i}:{j}]:{hex(_block[i][j])}")
        return _block

    def _import_plain_text(self, plain_text):
        self.__painText_list = plain_text
        return self.__convert_plain_text_to_block(self.__painText_list)

    def _convert_to_sub_byte(self, _16bit_block_text):
        for i in range(4):
            for j in range(4):
                _16bit_block_text[i][j] = self.__Sbox[_16bit_block_text[i][j] >> 4][_16bit_block_text[i][j] & 0x0F]
                # print(f"[{i}:{j}]:{hex(_16bit_block_text[i][j])}")
        return _16bit_block_text

    def _shit_row(self, _16bit_block_text):
        _shifted_raw = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            _shifted_raw[i] = np.roll(_16bit_block_text[i], -1*i)
        # for i in range(4):
        #     for j in range(4):
        #         print(f"[{i}][{j}]:{hex(_shifted_raw[i][j])}")
        return _shifted_raw

    def _mix_column(self, _16bit_block_text):
        GF_matrix = self.__GF(self.__mix_column_matrix)
        GF_Block = self.__GF(_16bit_block_text)
        result = np.dot(GF_matrix, GF_Block)
        result = np.array(result, dtype=np.uint8)
        # for i in range(4):
        #     for j in range(4):
        #         print(f"[{i}][{j}]:{hex(result[i][j])}")
        return result

    def _add_round_key(self, _16bit_block_text, _round_key):
        # for i in range(0, 4):
        #     print(f"----[{i}][{0}]:{hex(_16bit_block_text[i][0])}")
        # for i in range(0, 4):
        #     print(f"++++[{i}][{0}]:{hex(_round_key[i][0])}")
        my_result = _16bit_block_text[:, 0:1] ^ _round_key[:, 0:1]
        result = _16bit_block_text ^ _round_key
        for i in range(4):
            for j in range(4):
                print(f"[{i}][{j}]:{hex(result[i][j])}")
        return result


if __name__ == '__main__':
    myAE = AESEncrypt()
    around_key = np.array([[0xa0, 0x88, 0x23, 0x2a],
                           [0x9c, 0x54, 0xa3, 0x6c],
                           [0x7f, 0x2c, 0x39, 0x76],
                           [0xf2, 0xb1, 0x39, 0x05],
                           ], dtype=np.uint8)
    # 1:generate 4*4 block
    _16bit_block_text = myAE._import_plain_text((0x19, 0xa0, 0x9a, 0xe9,
                                                 0x3d, 0xf4, 0xC6, 0xf8,
                                                 0xe3, 0xe2, 0x8d, 0x48,
                                             0xbe, 0x2b, 0x2a, 0x08))
    # 2:generate sub bytes from Sbox
    _sub_bytes = myAE._convert_to_sub_byte(_16bit_block_text)
    # 3:shift row
    _shifted_raw = myAE._shit_row(_sub_bytes)
    # 4:mix column
    _mixed_column = myAE._mix_column(_shifted_raw)
    # 5:add round key
    _AddRoundKey = myAE._add_round_key(_sub_bytes, around_key)
    print(_AddRoundKey)
