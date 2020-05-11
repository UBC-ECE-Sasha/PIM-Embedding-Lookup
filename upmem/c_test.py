from ctypes import *
import numpy

so_file="/root/dlrm/emblib.so"

my_functions=CDLL(so_file)

#populates

my_functions.populate_mram.argtypes = c_uint8, c_uint32, c_uint32, c_uint32, POINTER(c_float)
my_functions.populate_mram.restype= None
p=(c_float*20)(1.0,2.0,3.0,4.0,5.0,2.0,4.0,6.0,8.0,10.0,3.0,6.0,9.0,12.0,15.0,4.0,8.0,12.0,16.0,20.0)
my_functions.populate_mram(0,1,4,5,p)

my_functions.populate_mram.argtypes = c_uint8, c_uint32, c_uint32, c_uint32, POINTER(c_float)
my_functions.populate_mram.restype= None
p=(c_float*20)(1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,4.0)
my_functions.populate_mram(1,1,4,5,p)

#lookups

my_functions.lookup.argtypes= c_uint8, POINTER(c_float), POINTER(c_uint32), c_uint32, c_uint32, c_uint32
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_float*5)(*tmp)
q=(c_uint32*1)(2)
my_functions.lookup(0,ans,q,1,5,4)

my_functions.lookup.argtypes= c_uint8, POINTER(c_float), POINTER(c_uint32), c_uint32, c_uint32, c_uint32
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_float*5)(*tmp)
q=(c_uint32*2)(3,0)
my_functions.lookup(1,ans,q,2,4,5)