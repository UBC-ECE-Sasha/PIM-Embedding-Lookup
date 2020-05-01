from ctypes import *
import numpy

so_file="/root/upmem/emblib.so"

my_functions=CDLL(so_file)

my_functions.populate_mram.argtypes = c_uint32, c_uint8, c_uint8, POINTER(c_float)
my_functions.populate_mram.restype= None
p=(c_float*20)(1.0,2.0,3.0,4.0,5.0,2.0,4.0,6.0,8.0,10.0,3.0,6.0,9.0,12.0,15.0,4.0,8.0,12.0,16.0,20.0)
my_functions.populate_mram(1,4,5,p)
my_functions.lookup.argtypes= POINTER(c_uint32), c_uint8, c_uint8, c_uint8
my_functions.lookup.restype= None
q=(c_uint32*4)(1,2,0,3)
my_functions.lookup(q,4,5,4)
