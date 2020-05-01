from ctypes import *
import numpy

so_file="/root/upmem/emblib.so"

my_functions=CDLL(so_file)

my_functions.populate_mram.argtypes = c_uint32, c_uint8, c_uint8, POINTER(c_uint32)
my_functions.populate_mram.restype= None
p=(c_uint32*20)()
my_functions.populate_mram(1,4,5,p)
#set_pointer=my_functions.populate_mram(1,4, 5, {1,2,3,4,5,2,4,6,8,10,3,6,9,12,15,4,8,12,16,20})
my_functions.lookup.argtypes= POINTER(c_uint32), c_uint8, c_uint8, c_uint8
my_functions.lookup.restype= None
q=(c_uint32*4)()
my_functions.lookup(q,4,5,4)

