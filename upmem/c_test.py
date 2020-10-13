from ctypes import *
import numpy

so_file="/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/emblib.so"

my_functions=CDLL(so_file)

#populates

my_functions.populate_mram.argtypes = c_uint32, c_uint64, c_uint64, POINTER(c_int32)
my_functions.populate_mram.restype= None
data_ptr=(c_int32*20)(1,2,3,4,5,2,4,6,8,10,3,6,9,12,15,4,8,12,16,20)
my_functions.populate_mram(0,4,5,data_ptr)

#lookups

""" my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_double*20)(*tmp)
q=(c_uint64*4)(0,1,2,3)
my_functions.lookup(ans,q,4,4,5) """
