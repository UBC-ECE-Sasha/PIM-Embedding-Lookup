.PHONY:all

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=64
NR_BATCHES=32
NR_RUN=1
INDEX_PER_BATCH=120
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=500000
NR_EMBEDDING=500
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0


build: build/emb build/embdpu

build/emb: src/*.c include/*.h
	mkdir -p build
	gcc -O3     -Wall -mavx -msse4 -lm -lz -lpthread -I include \
		-DNR_RUN=${NR_RUN} \
		-DCHECK_RESULTS=${CHECK_RESULTS} \
		-DMAX_NR_DPUS=${MAX_NR_DPUS} \
		-DRAND_INPUT_SIZE=${RAND_INPUT_SIZE} \
		-DMAX_INDEX_PER_BATCH_RAND=${MAX_INDEX_PER_BATCH_RAND} \
		-DNR_TASKLETS=${NR_TASKLETS} \
		-DMAX_INDEX_PER_BATCH=${MAX_INDEX_PER_BATCH} \
		-DMAX_NR_EMBEDDING=${MAX_NR_EMBEDDING} \
		-DMAX_NR_BATCHES=${MAX_NR_BATCHES} \
		-DINDEX_PER_BATCH=${INDEX_PER_BATCH} \
		-DNR_ROWS=${NR_ROWS} \
		-DNR_BATCHES=${NR_BATCHES} \
		-DNR_EMBEDDING=${NR_EMBEDDING} \
		-DNR_COLS=${NR_COLS} \
		-o build/emb src/*.c  `dpu-pkg-config --cflags --libs dpu` 

build/embdpu: src/dpu/*.c 
	mkdir -p build
	# TODO : why flto fails
	dpu-clang -O3  -flto=thin -I include \
		-DNR_RUN=${NR_RUN} \
		-DNR_TASKLETS=${NR_TASKLETS} \
		-DMAX_NR_DPUS=${MAX_NR_DPUS} \
		-DRAND_INPUT_SIZE=${RAND_INPUT_SIZE} \
		-DMAX_INDEX_PER_BATCH=${MAX_INDEX_PER_BATCH} \
		-DMAX_NR_EMBEDDING=${MAX_NR_EMBEDDING} \
		-DMAX_NR_BATCHES=${MAX_NR_BATCHES} \
		-DNR_EMBEDDING=${NR_EMBEDDING} \
		-DINDEX_PER_BATCH=${INDEX_PER_BATCH} \
		-DNR_ROWS=${NR_ROWS} \
		-DNR_BATCHES=${NR_BATCHES} \
		-o build/embdpu src/dpu/dpu_embedding.c

clean:
	([ -d "build" ] && rm -r build/ && rm -f out.cpu.log out.dpu.log) || [ ! -d "build" ]

testdpu: FORCE
	./build/emb

tracedpu: FORCE
	dpu-profiling functions -o dpu.json -a -A \
		-f gather_rank_embedding_results \
		-f build_synthetic_input_data \
		-f synthetic_inference \
		-f populate_mram \
		-f lookup \
		-- ./build/emb

FORCE: ;
