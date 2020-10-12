from ctypes import *
import numpy

so_file="emblib.so"

my_functions=CDLL(so_file)

#populates

my_functions.populate_mram.argtypes = c_uint32, c_uint64, c_uint64, POINTER(c_int32)
my_functions.populate_mram.restype= None
data_ptr=(c_int32*20)(1.0,2.0,3.0,4.0,5.0,2.0,4.0,6.0,8.0,10.0,3.0,6.0,9.0,12.0,15.0,4.0,8.0,12.0,16.0,20.0)
my_functions.populate_mram(0,4,5,data_ptr)

#lookups

""" my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_double*20)(*tmp)
q=(c_uint64*4)(0,1,2,3)
my_functions.lookup(ans,q,4,4,5) """
