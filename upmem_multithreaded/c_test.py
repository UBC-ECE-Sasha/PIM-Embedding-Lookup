from ctypes import *
import numpy

so_file="/home/upmem0016/abbask/multi/emblib.so"

my_functions=CDLL(so_file)


#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_double)
my_functions.populate_mram.restype= None

l = []

#1000*4 table
for i in range (0, 32):
    for j in range(0, 4):
        l.append(c_double(i))

p=(c_double*(32*4))(*l)
my_functions.populate_mram(32,4,p)

#lookups

my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0 for i in range(0, 40)]
ans=(c_double*40)(*tmp)

q=(c_uint64*10)(0,1,2,3,4,5,6,7,8,9)
my_functions.lookup(ans,q,10,32,4)




"""
#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_double)
my_functions.populate_mram.restype= None
p=(c_double*20)(1.0,2.0,3.0,4.0,5.0,2.0,4.0,6.0,8.0,10.0,3.0,6.0,9.0,12.0,15.0,4.0,8.0,12.0,16.0,20.0)
my_functions.populate_mram(4,5,p)

#lookups

my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ans=(c_double*20)(*tmp)

q=(c_uint64*4)(0,1,2,3)
my_functions.lookup(ans,q,4,4,5)




#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_double)
my_functions.populate_mram.restype= None

l = []

#1000*4 table
for i in range (0, 32):
    for j in range(0, 4):
        l.append(c_double(i))

p=(c_double*(32*4))(*l)
my_functions.populate_mram(32,4,p)

#lookups

my_functions.lookup.argtypes= POINTER(c_double), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None
tmp=[0.0 for i in range(0, 36)]
ans=(c_double*40)(*tmp)

q=(c_uint64*10)(0,1,2,3,4,5,6,7,8,9)
my_functions.lookup(ans,q,10,32,4)

"""