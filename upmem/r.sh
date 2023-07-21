#!/bin/sh
# ./run.sh -br random
# export MAX_NR_BATCHES=8
# nohup ./run.sh -br random > batch_8.out

export NR_COLS=64
export MAX_NR_BATCHES=64
export NR_TASKLETS=14
export MAX_INDICES_PER_BATCH=120
export NR_TABLES=32

# ONE TIME
# export TABLE_CONFIG=
# make clean
# make
# ./run-build.sh -br random

# TABLE SIZE EXP
# make clean
# make
# export TABLE_CONFIG=2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000
export TABLE_CONFIG=125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000-125000
# make clean
# make
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize125.out
# export TABLE_CONFIG=250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize250.out
# export TABLE_CONFIG=500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize500.out
# export TABLE_CONFIG=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize1000.out
# export TABLE_CONFIG=2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000-2000000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize2000.out
# export TABLE_CONFIG=4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000-4000000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize4000.out
# export TABLE_CONFIG=8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000-8000000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize8000.out
# export TABLE_CONFIG=13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000-13900000
# nohup ./run.sh -br random > ./post_opt_res/DPUTableSize13900.out

# NUM TABLES EXP
# export NR_TABLES=32
# export TABLE_CONFIG=500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000
# make clean
# make
# nohup ./run-build.sh -br random > ./post_opt_res/DPUNumTables32-1.out
# export NR_TABLES=16
# export TABLE_CONFIG=500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000
# make clean
# make
# nohup ./run-build.sh -br random > ./post_opt_res/DPUNumTables16-1.out
# export NR_TABLES=8
# export TABLE_CONFIG=500000-500000-500000-500000-500000-500000-500000-500000
# make clean
# make
# nohup ./run-build.sh -br random > ./post_opt_res/DPUNumTables8-1.out
# export NR_TABLES=4
# export TABLE_CONFIG=500000-500000-500000-500000
# make clean
# make
# nohup ./run-build.sh -br random > ./post_opt_res/DPUNumTables4-1.out
# export NR_TABLES=2
# export TABLE_CONFIG=500000-500000
# make clean
# make
# nohup ./run-build.sh -br random > ./post_opt_res/DPUNumTables2-1.out

# BATCH SIZE EXP
# export TABLE_CONFIG=500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000
# export MAX_NR_BATCHES=8
# make clean
# make
# nohup ./run-build.sh -br random > CPUBatch_8.out
# export MAX_NR_BATCHES=16
# make clean
# make
# nohup ./run.sh -br random > CPUBatch_16.out
# export MAX_NR_BATCHES=32
# make clean
# make
# nohup ./run.sh -br random > CPUBatch_32.out
# export MAX_NR_BATCHES=64
# make clean
# make
# nohup ./run.sh -br random > CPUBatch_64.out
# export MAX_NR_BATCHES=100
# make clean
# make
./run.sh -br random #> CPUBatch_100.out