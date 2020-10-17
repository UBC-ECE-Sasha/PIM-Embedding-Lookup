import argparse
import os
from ctypes import *

so_file = None
my_functions = None

# Defaults, can be overriden by cli arguments
if __name__ != "__main__":
    so_file = "./emblib.so"
    my_functions = CDLL(os.path.abspath(so_file))


def parse():
    parser = argparse.ArgumentParser(description="Test embedding")

    parser.add_argument(
        "--so_file", help="Input DLL", default="./emblib.so", type=str
    )

    return parser.parse_args()


def populate():
    my_functions.populate_mram.argtypes = c_uint32, c_uint64, c_uint64, POINTER(c_int32)
    my_functions.populate_mram.restype = None
    data_ptr = (c_int32 * 20)(1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20)
    my_functions.populate_mram(0, 4, 5, data_ptr)
    my_functions.populate_mram(2, 4, 5, data_ptr)
    my_functions.populate_mram(3, 4, 5, data_ptr)
    my_functions.populate_mram(4, 4, 5, data_ptr)
    my_functions.populate_mram(5, 4, 5, data_ptr)
    my_functions.populate_mram(6, 4, 5, data_ptr)
    my_functions.populate_mram(7, 4, 5, data_ptr)


def lookup():
    my_functions.lookup.argtypes = POINTER(c_double), POINTER(
        c_uint64), c_uint64, c_uint64, c_uint64
    my_functions.lookup.restype = None
    tmp = [[0.0] * 20]
    ans = (c_double * 20)(*tmp)
    q = (c_uint64 * 4)(0, 1, 2, 3)
    my_functions.lookup(ans, q, 4, 4, 5)


if __name__ == "__main__":
    config = parse()
    so_file = config.so_file
    my_functions = CDLL(os.path.abspath(so_file))

    populate()

    # TODO
    # lookup()
