# perf multicol

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=128
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=160
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=700000
NR_EMBEDDING=200
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 20, dpu part col 8
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 20
nr_dpus 1400
nr cols per dpu 20
alloc dpus 1400
generate synthetic tables

# multirun 20
dpu [ms]: 14833.226350, cpu [ms] 82837.979600, dpu acceleration 5.584623
 DPU PRATIO 692.852194, CPU PRATIO 2885.025300, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


#multirun 1
[PERF] CPU CLOCK RAW time [ms]: 248260.27
[PERF] CPUTIME time [ms]: 2663050.24
[PERF] CPU PROCESS ratio: 1072.68
dpu [ms]: 20369.388000, cpu [ms] 248260.272000, dpu acceleration 12.187910
 DPU PRATIO 886.711216, CPU PRATIO 1072.684815, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)





MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=32
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=160
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=700000
NR_EMBEDDING=200
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

nr cols per dpu 20
alloc dpus 400
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 33600000 nr cols 12
start inference
max nr embedding 2000
dpu [ms]: 17311.540000, cpu [ms] 231748.224000, dpu acceleration 13.386921
 DPU PRATIO 393.474873, CPU PRATIO 994.756240, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)




MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=64
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=160
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=700000
NR_EMBEDDING=500
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0
min nr cols per dpu 2
nr cols per dpus 20, dpu part col 4
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 20
nr_dpus 2000
nr cols per dpu 20
alloc dpus 2000
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 11200000 nr cols 4
start inference
max nr embedding 2000

dpu [ms]: 24410.542000, cpu [ms] 332933.120000, dpu acceleration 13.638907
 DPU PRATIO 1138.050339, CPU PRATIO 1058.865454, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)






# perf DPU
        cyc       inst    ratio
all     2530256  2488080  0.98
init    28240    1808     0.064



# update Mt (data non generated falsly boost CPU)

# emb data pure rand (RAND)
max nr embedding 2000
dpu [ms]: 18053.449000, cpu [ms] 127023.148800, dpu acceleration 7.035949
 DPU PRATIO 855.690423, CPU PRATIO 3077.237286, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


# emb data counter (CPT) 
max nr embedding 2000
dpu [ms]: 18862.684700, cpu [ms] 152838.430000, dpu acceleration 8.102687
 DPU PRATIO 857.428834, CPU PRATIO 2926.055898, DPU OK ? 1 

# emb data non init (NONE)
dpu [ms]: 17877.672750, cpu [ms] 18041.412150, dpu acceleration 1.009159
 DPU PRATIO 932.438170, CPU PRATIO 2622.523159, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)



# update Mt(fix CPU mt bug), cloud7

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=16
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=160
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=700000
NR_EMBEDDING=150
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0


max nr embedding 2000
dpu [ms]: 18901.936400, cpu [ms] 17462.926300, dpu acceleration 0.923870
 DPU PRATIO 865.897869, CPU PRATIO 2679.371648, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)











# 2 (cpu multitherad)
MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=16
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=120
MAX_INDEX_PER_BATCH_RAND=100
NR_ROWS=700000
NR_EMBEDDING=2000
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 16, dpu part col 0
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 16
nr_dpus 2000
nr cols per dpu 16
alloc dpus 2000
populate mram with embedding synthetic tables
start inference
max nr embedding 2000
dpu [ms]: 24.325885, cpu [ms] 807.520337, dpu acceleration 33.195928, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

# 2 (cpu singlethread)
./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 16, dpu part col 0
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 16
nr_dpus 2000
nr cols per dpu 16
alloc dpus 2000
populate mram with embedding synthetic tables
start inference
max nr embedding 2000
dpu [ms]: 25.756069, cpu [ms] 3244.451591, dpu acceleration 125.968431, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)





# 1
MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=16
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=120
MAX_INDEX_PER_BATCH_RAND=100
NR_ROWS=700000
NR_EMBEDDING=2000
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0
./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
nr cols per dpus 16, dpu part col 0
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 16
nr_dpus 2000
alloc dpus 2000
populate mram with embedding synthetic tables
start inference
max nr embedding 2000
dpu [ms]: 22.364698, cpu [ms] 1184.088887, dpu acceleration 52.944551, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)




# 0 
MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=1000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=16
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=120
MAX_INDEX_PER_BATCH_RAND=100
NR_ROWS=700000
NR_EMBEDDING=1000
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
nr cols per dpus 16, dpu part col 0
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 16
nr_dpus 1000
alloc dpus 1000
populate mram with embedding synthetic tables
start inference
max nr embedding 1000
dpu [ms]: 14.792739, cpu [ms] 581.375319, dpu acceleration 39.301398, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

# 1

















cloud8
MAX_NR_EMBEDDING=100
MAX_INDEX_PER_BATCH=128
MAX_NR_BATCHES=64
NR_COLS=64
NR_BATCHES=32
NR_RUN=100
INDEX_PER_BATCH=128
MAX_INDEX_PER_BATCH_RAND=100
NR_ROWS=14000000
NR_EMBEDDING=30
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0
alloc dpus 1920 
alloc nr_dpus 1920
Alloc DPUs,  NR_RANKS 31
rank 0 nr dpus 63
rank 1 nr dpus 64
rank 2 nr dpus 62
rank 3 nr dpus 63
rank 4 nr dpus 63
rank 5 nr dpus 56
rank 6 nr dpus 63
rank 7 nr dpus 64
rank 8 nr dpus 64
rank 9 nr dpus 64
rank 10 nr dpus 64
rank 11 nr dpus 64
rank 12 nr dpus 64
rank 13 nr dpus 58
rank 14 nr dpus 56
rank 15 nr dpus 64
rank 16 nr dpus 64
rank 17 nr dpus 62
rank 18 nr dpus 56
rank 19 nr dpus 64
rank 20 nr dpus 64
rank 21 nr dpus 64
rank 22 nr dpus 64
rank 23 nr dpus 64
rank 24 nr dpus 56
rank 25 nr dpus 62
rank 26 nr dpus 64
rank 27 nr dpus 64
rank 28 nr dpus 56
rank 29 nr dpus 56
rank 30 nr dpus 64

dpu [ms]: 3.792112, cpu [ms] 163.137815, dpu acceleration 43.020305, DPU OK ? 1 
dpu [ms]: 4.043170, cpu [ms] 143.384237, dpu acceleration 35.463321, DPU OK ? 1 
dpu [ms]: 3.803050, cpu [ms] 129.167752, dpu acceleration 33.964253, DPU OK ? 1 
dpu [ms]: 3.760563, cpu [ms] 140.303063, dpu acceleration 37.309058, DPU OK ? 1 
dpu [ms]: 3.903051, cpu [ms] 133.385881, dpu acceleration 34.174773, DPU OK ? 1 
dpu [ms]: 3.509724, cpu [ms] 127.775989, dpu acceleration 36.406278, DPU OK ? 1 
dpu [ms]: 3.954819, cpu [ms] 139.444978, dpu acceleration 35.259509, DPU OK ? 1 
dpu [ms]: 3.612390, cpu [ms] 134.544759, dpu acceleration 37.245358, DPU OK ? 1 
dpu [ms]: 3.605661, cpu [ms] 137.922531, dpu acceleration 38.251663, DPU OK ? 1 
dpu [ms]: 3.414589, cpu [ms] 126.193990, dpu acceleration 36.957300, DPU OK ? 1 
dpu [ms]: 3.532768, cpu [ms] 131.471807, dpu acceleration 37.214956, DPU OK ? 1 
dpu [ms]: 3.867978, cpu [ms] 138.602536, dpu acceleration 35.833331, DPU OK ? 1 
dpu [ms]: 3.764971, cpu [ms] 125.864143, dpu acceleration 33.430309, DPU OK ? 1 
dpu [ms]: 3.771465, cpu [ms] 130.859990, dpu acceleration 34.697389, DPU OK ? 1 
dpu [ms]: 3.877276, cpu [ms] 131.261223, dpu acceleration 33.853980, DPU OK ? 1 
dpu [ms]: 3.680324, cpu [ms] 129.913026, dpu acceleration 35.299345, DPU OK ? 1 
dpu [ms]: 3.739265, cpu [ms] 124.174293, dpu acceleration 33.208209, DPU OK ? 1 
dpu [ms]: 3.870194, cpu [ms] 136.271004, dpu acceleration 35.210381, DPU OK ? 1 
dpu [ms]: 3.925622, cpu [ms] 123.764071, dpu acceleration 31.527251, DPU OK ? 1 
dpu [ms]: 11.415245, cpu [ms] 125.310495, dpu acceleration 10.977469, DPU OK ? 1 
dpu [ms]: 3.755615, cpu [ms] 135.795115, dpu acceleration 36.157890, DPU OK ? 1 
dpu [ms]: 3.974138, cpu [ms] 127.376660, dpu acceleration 32.051393, DPU OK ? 1 
dpu [ms]: 3.863569, cpu [ms] 123.095833, dpu acceleration 31.860653, DPU OK ? 1 
dpu [ms]: 3.876038, cpu [ms] 125.214258, dpu acceleration 32.304703, DPU OK ? 1 
dpu [ms]: 3.963908, cpu [ms] 135.159629, dpu acceleration 34.097570, DPU OK ? 1 
dpu [ms]: 3.779182, cpu [ms] 122.910528, dpu acceleration 32.523051, DPU OK ? 1 
dpu [ms]: 3.880123, cpu [ms] 133.221248, dpu acceleration 34.334285, DPU OK ? 1 
dpu [ms]: 3.955883, cpu [ms] 68.070226, dpu acceleration 17.207341, DPU OK ? 1 
dpu [ms]: 3.341139, cpu [ms] 70.442602, dpu acceleration 21.083410, DPU OK ? 1 
dpu [ms]: 4.592526, cpu [ms] 72.564336, dpu acceleration 15.800528, DPU OK ? 1 
dpu [ms]: 3.463781, cpu [ms] 69.876803, dpu acceleration 20.173563, DPU OK ? 1 
dpu [ms]: 3.659296, cpu [ms] 63.945310, dpu acceleration 17.474757, DPU OK ? 1 
dpu [ms]: 3.677162, cpu [ms] 75.201677, dpu acceleration 20.451010, DPU OK ? 1 
dpu [ms]: 3.335860, cpu [ms] 69.573077, dpu acceleration 20.856114, DPU OK ? 1 
dpu [ms]: 11.379984, cpu [ms] 69.678429, dpu acceleration 6.122893, DPU OK ? 1 
dpu [ms]: 3.769593, cpu [ms] 68.786589, dpu acceleration 18.247750, DPU OK ? 1 
dpu [ms]: 3.392237, cpu [ms] 63.039890, dpu acceleration 18.583575, DPU OK ? 1 
dpu [ms]: 3.604687, cpu [ms] 62.815506, dpu acceleration 17.426064, DPU OK ? 1 
dpu [ms]: 3.510495, cpu [ms] 70.068809, dpu acceleration 19.959809, DPU OK ? 1 
dpu [ms]: 3.330445, cpu [ms] 74.478037, dpu acceleration 22.362788, DPU OK ? 1 
dpu [ms]: 3.637376, cpu [ms] 62.344608, dpu acceleration 17.139995, DPU OK ? 1 
dpu [ms]: 3.461730, cpu [ms] 62.128424, dpu acceleration 17.947218, DPU OK ? 1 
dpu [ms]: 3.578112, cpu [ms] 67.782026, dpu acceleration 18.943517, DPU OK ? 1 
dpu [ms]: 3.821712, cpu [ms] 62.147615, dpu acceleration 16.261721, DPU OK ? 1 
dpu [ms]: 3.554010, cpu [ms] 62.193652, dpu acceleration 17.499571, DPU OK ? 1 
dpu [ms]: 10.706197, cpu [ms] 64.379790, dpu acceleration 6.013320, DPU OK ? 1 
dpu [ms]: 3.629177, cpu [ms] 68.312460, dpu acceleration 18.823127, DPU OK ? 1 
dpu [ms]: 3.590345, cpu [ms] 69.675117, dpu acceleration 19.406246, DPU OK ? 1 
dpu [ms]: 3.539965, cpu [ms] 66.562587, dpu acceleration 18.803177, DPU OK ? 1 
dpu [ms]: 3.712058, cpu [ms] 70.318931, dpu acceleration 18.943382, DPU OK ? 1 
dpu [ms]: 3.517493, cpu [ms] 62.628211, dpu acceleration 17.804786, DPU OK ? 1 
dpu [ms]: 3.516381, cpu [ms] 74.151083, dpu acceleration 21.087329, DPU OK ? 1 
dpu [ms]: 3.398716, cpu [ms] 62.020428, dpu acceleration 18.248194, DPU OK ? 1 
dpu [ms]: 3.549116, cpu [ms] 68.312622, dpu acceleration 19.247785, DPU OK ? 1 
dpu [ms]: 3.711894, cpu [ms] 61.889660, dpu acceleration 16.673337, DPU OK ? 1 
dpu [ms]: 3.546434, cpu [ms] 69.105926, dpu acceleration 19.486032, DPU OK ? 1 
dpu [ms]: 3.474864, cpu [ms] 67.951330, dpu acceleration 19.555105, DPU OK ? 1 
dpu [ms]: 3.529826, cpu [ms] 60.880337, dpu acceleration 17.247405, DPU OK ? 1 
dpu [ms]: 3.508891, cpu [ms] 68.618196, dpu acceleration 19.555522, DPU OK ? 1 
dpu [ms]: 3.493259, cpu [ms] 67.936748, dpu acceleration 19.447956, DPU OK ? 1 
dpu [ms]: 3.691129, cpu [ms] 60.547824, dpu acceleration 16.403606, DPU OK ? 1 
dpu [ms]: 5.151405, cpu [ms] 60.403613, dpu acceleration 11.725658, DPU OK ? 1 
dpu [ms]: 3.593467, cpu [ms] 60.300746, dpu acceleration 16.780659, DPU OK ? 1 
dpu [ms]: 4.112866, cpu [ms] 59.857943, dpu acceleration 14.553828, DPU OK ? 1 
dpu [ms]: 4.299801, cpu [ms] 67.373321, dpu acceleration 15.668939, DPU OK ? 1 
dpu [ms]: 3.509549, cpu [ms] 59.258741, dpu acceleration 16.885002, DPU OK ? 1 
dpu [ms]: 3.552055, cpu [ms] 59.442825, dpu acceleration 16.734770, DPU OK ? 1 
dpu [ms]: 4.458851, cpu [ms] 65.201385, dpu acceleration 14.622912, DPU OK ? 1 
dpu [ms]: 3.489958, cpu [ms] 58.989411, dpu acceleration 16.902613, DPU OK ? 1 
dpu [ms]: 3.658575, cpu [ms] 58.707952, dpu acceleration 16.046672, DPU OK ? 1 
dpu [ms]: 3.775020, cpu [ms] 58.575210, dpu acceleration 15.516530, DPU OK ? 1 
dpu [ms]: 3.448895, cpu [ms] 58.760395, dpu acceleration 17.037455, DPU OK ? 1 
dpu [ms]: 3.668629, cpu [ms] 66.361266, dpu acceleration 18.088846, DPU OK ? 1 
dpu [ms]: 3.859426, cpu [ms] 58.928441, dpu acceleration 15.268706, DPU OK ? 1 
dpu [ms]: 3.412970, cpu [ms] 64.791139, dpu acceleration 18.983800, DPU OK ? 1 
dpu [ms]: 3.550839, cpu [ms] 57.730755, dpu acceleration 16.258342, DPU OK ? 1 
dpu [ms]: 3.823713, cpu [ms] 64.987545, dpu acceleration 16.995926, DPU OK ? 1 
dpu [ms]: 3.361535, cpu [ms] 64.939345, dpu acceleration 19.318361, DPU OK ? 1 
dpu [ms]: 3.517359, cpu [ms] 57.800170, dpu acceleration 16.432832, DPU OK ? 1 
dpu [ms]: 4.467995, cpu [ms] 64.362036, dpu acceleration 14.405127, DPU OK ? 1 
dpu [ms]: 3.491235, cpu [ms] 57.166059, dpu acceleration 16.374165, DPU OK ? 1 
dpu [ms]: 3.282369, cpu [ms] 56.695181, dpu acceleration 17.272641, DPU OK ? 1 
dpu [ms]: 3.324095, cpu [ms] 56.773918, dpu acceleration 17.079511, DPU OK ? 1 
dpu [ms]: 3.414872, cpu [ms] 56.535048, dpu acceleration 16.555539, DPU OK ? 1 
dpu [ms]: 3.358672, cpu [ms] 56.507791, dpu acceleration 16.824445, DPU OK ? 1 
dpu [ms]: 3.653956, cpu [ms] 56.252328, dpu acceleration 15.394911, DPU OK ? 1 
dpu [ms]: 3.451347, cpu [ms] 56.096112, dpu acceleration 16.253397, DPU OK ? 1 
dpu [ms]: 3.577768, cpu [ms] 55.692037, dpu acceleration 15.566140, DPU OK ? 1 
dpu [ms]: 3.444286, cpu [ms] 55.915629, dpu acceleration 16.234316, DPU OK ? 1 
dpu [ms]: 3.550622, cpu [ms] 55.606490, dpu acceleration 15.661056, DPU OK ? 1 
dpu [ms]: 3.417586, cpu [ms] 55.472428, dpu acceleration 16.231465, DPU OK ? 1 
dpu [ms]: 3.509897, cpu [ms] 62.346607, dpu acceleration 17.763087, DPU OK ? 1 
dpu [ms]: 3.526847, cpu [ms] 55.481194, dpu acceleration 15.731103, DPU OK ? 1 
dpu [ms]: 3.743272, cpu [ms] 61.701135, dpu acceleration 16.483209, DPU OK ? 1 
dpu [ms]: 3.639061, cpu [ms] 55.265585, dpu acceleration 15.186771, DPU OK ? 1 
dpu [ms]: 3.301033, cpu [ms] 55.083287, dpu acceleration 16.686682, DPU OK ? 1 
dpu [ms]: 3.347292, cpu [ms] 54.866029, dpu acceleration 16.391169, DPU OK ? 1 
dpu [ms]: 3.509053, cpu [ms] 55.169426, dpu acceleration 15.722027, DPU OK ? 1 
dpu [ms]: 3.503731, cpu [ms] 61.923099, dpu acceleration 17.673474, DPU OK ? 1 
dpu [ms]: 4.547925, cpu [ms] 54.496252, dpu acceleration 11.982663, DPU OK ? 1 

