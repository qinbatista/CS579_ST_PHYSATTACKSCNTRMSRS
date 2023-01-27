Hi everyone,

Homework 2 is now available on Canvas. This time, there is no dedicated Jupyter template, as reusing the template from homework 1 is possible.

How to read the files in Python/Jupyter:

Task 1: data = np.fromfile('measurement_data_2023_uint8.bin',dtype='uint8'); data.shape will then give you: (1000000000,); reading like this is OK -- for processing, only incremental steps (byte-after-byte)
Task 2/3: too bad, Canvas does not seem to offer code-formatting ...
all_plaintexts_unstructured=np.fromfile('plaintext_10000x16_uint8.bin', dtype='uint8')
plaintext_10000_16 = np.reshape(all_plaintexts_unstructured,(-1,16))
firstbyte = plaintext_10000_16[:,0]
traces=np.fromfile('./traces_10000x50_int8.bin', dtype='int8')
traces = np.reshape(traces,(-1,50))
firstbyte.shape
gives you (10000,)
traces.shape
gives you (10000, 50)
Throughout the next classes, we will proceed as before. I will check your progress, and provide reasonable help.
Have fun,
Vincent