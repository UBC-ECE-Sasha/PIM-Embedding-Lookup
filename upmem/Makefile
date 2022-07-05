.PHONY:all

MAX_INDEX_PER_BATCH=32
MAX_NR_BATCHES=64
NR_COLS=8
NR_BATCHES=1
NR_RUN=1
INDEX_PER_BATCH=32
NR_ROWS=16
NR_EMBEDDING=1
NR_TASKLETS=16


build: build/emb build/embdpu

build/emb: src/*.c include/*.h
	mkdir -p build
	gcc -O0 -g -Wall -msse4 -lm -lz -lpthread -I include -DNR_RUN=${NR_RUN} -DNR_TASKLETS=${NR_TASKLETS} -DMAX_INDEX_PER_BATCH=${MAX_INDEX_PER_BATCH}  -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DMAX_NR_BATCHES=${MAX_NR_BATCHES} -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DNR_ROWS=${NR_ROWS} -DNR_BATCHES=${NR_BATCHES} -DNR_EMBEDDING=${NR_EMBEDDING}  -DNR_COLS=${NR_COLS} -o build/emb src/*.c  `dpu-pkg-config --cflags --libs dpu` 

build/embdpu: src/dpu/*.c 
	mkdir -p build
	# TODO : why flto fails
	dpu-clang -g0 -O3 -flto=thin  -I include -DNR_RUN=${NR_RUN} -DNR_TASKLETS=${NR_TASKLETS} -DMAX_INDEX_PER_BATCH=${MAX_INDEX_PER_BATCH}  -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DMAX_NR_BATCHES=${MAX_NR_BATCHES}  -DNR_EMBEDDING=${NR_EMBEDDING} -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DNR_ROWS=${NR_ROWS}   -DNR_BATCHES=${NR_BATCHES} -o build/embdpu src/dpu/dpu_embedding.c

clean:
	([ -d "build" ] && rm -r build/ && rm -f out.cpu.log out.dpu.log) || [ ! -d "build" ]

testdpu: FORCE
	./build/emb

tracedpu: FORCE
	dpu-profiling functions -o dpu.json -a -A -f synthetic_inference -f populate_mram -f lookup -- ./build/emb

FORCE: ;
