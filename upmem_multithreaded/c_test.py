from ctypes import *
import numpy

so_file="/home/upmem0016/abbask/multi/emblib.so"

my_functions=CDLL(so_file)


#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_double)
my_functions.populate_mram.restype= None

l = []

for i in range (0, 1000):
    for j in range(0, 16):
        l.append(c_double(i))

p=(c_double*(1000*16))(*l)
my_functions.populate_mram(1000,16,p)

#lookups

my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None

tmp1=[0.0 for i in range(0, 1000)]
ans=(c_double*1000)(*tmp1)

tmp2=[i for i in range(0, 250)]
q=(c_uint64*250)(*tmp2)

my_functions.lookup(ans,q,250,1000,16)