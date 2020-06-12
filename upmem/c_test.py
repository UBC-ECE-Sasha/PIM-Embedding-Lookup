from ctypes import *
import numpy

so_file="/home/upmem0016/abbask/emb/emblib.so"

my_functions=CDLL(so_file)

#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_double)
my_functions.populate_mram.restype= None
p=(c_double*20)(1.0,2.0,3.0,4.0,5.0,2.0,4.0,6.0,8.0,10.0,3.0,6.0,9.0,12.0,15.0,4.0,8.0,12.0,16.0,20.0)
my_functions.populate_mram(4,5,p)

#lookups

my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_double*5)(*tmp)
q=(c_uint64*3)(0,2,1,)
my_functions.lookup(ans,q,3,4,5)
