from ctypes import *
import numpy
import time

so_file="/home/upmem0016/abbask/multicore/emblib.so"

my_functions=CDLL(so_file)


#populates

my_functions.populate_mram.argtypes = c_uint64, c_uint64, POINTER(c_uint64)
my_functions.populate_mram.restype= None

l = []

for i in range (0, 1000):
    for j in range(0, 16):
        l.append(c_uint64(i))

p=(c_uint64*(1000*16))(*l)
my_functions.populate_mram(1000, 16, p)

#lookups

my_functions.lookup.argtypes= POINTER(c_uint64), POINTER(c_uint64), c_uint64, c_uint64, c_uint64
my_functions.lookup.restype= None

tmp1=[0 for i in range(0, 128)]
ans=(c_uint64*128)(*tmp1)

tmp2=[i for i in range(0, 8)]
q=(c_uint64*8)(*tmp2)

begin_time = time.time()

my_functions.lookup(ans, q, 8, 1000, 16)

end_time = time.time()

print("time elapsed: ", end_time-begin_time)