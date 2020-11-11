import argparse
import os
from ctypes import *

so_file = None
my_functions = None
num_dpu= 8
# Defaults, can be overriden by cli arguments
if __name__ != "__main__":
    so_file = "./emblib.so"
    my_functions = CDLL(os.path.abspath(so_file))
    num_dpu= 8


def parse():
    parser = argparse.ArgumentParser(description="Test embedding")

    parser.add_argument(
        "--so_file", help="Input DLL", default="./emblib.so", type=str)
    parser.add_argument(
        "--num_dpu", help="Input number of DPUs", default=8, type=int)

    return parser.parse_args()


def populate():
    my_functions.populate_mram.argtypes = c_uint32, c_uint64, POINTER(c_int32)
    my_functions.populate_mram.restype = None
    data_ptr = (c_int32 * 20)(1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20)
    for i in range (0, num_dpu):
        my_functions.populate_mram(i, 4, data_ptr)


def lookup():
    my_functions.lookup.argtypes = POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint32)
    my_functions.lookup.restype = None
    indices=(c_uint32 * 24)(1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2)
    offsets=(c_uint32 * 16)(0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2)
    indices_len=(c_uint64 * 8)(3,3,3,3,3,3,3,3)
    offsets_len=(c_uint64 * 8)(2,2,2,2,2,2,2,2)
    ans=(c_uint32 *20)()
    my_functions.lookup(indices, offsets, indices_len, offsets_len,ans)


if __name__ == "__main__":
    config = parse()
    so_file = config.so_file
    my_functions = CDLL(os.path.abspath(so_file))

    populate()
    lookup()
