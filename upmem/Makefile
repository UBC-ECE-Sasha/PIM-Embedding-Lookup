.PHONY:all

NR_COLS=16
NR_BATCHES=2
NR_RUN=2
INDEX_PER_BATCH=32
NR_ROWS=32
NR_EMBEDDING=1

build: build/emb build/embdpu

build/emb: src/*.c include/*.h
	mkdir -p build
	gcc -O3 -g -Wall -msse4 -lm -lz -lpthread -I include -DNR_RUN=${NR_RUN} -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DNR_ROWS=${NR_ROWS} -DNR_BATCHES=${NR_BATCHES} -DNR_EMBEDDING=${NR_EMBEDDING}  -DNR_COLS=${NR_COLS} -o build/emb src/*.c  `dpu-pkg-config --cflags --libs dpu` 

build/embdpu: src/dpu/*.c 
	mkdir -p build
	# TODO : why flto fails
	dpu-clang -g0 -O3 -flto=thin  -I include  -DNR_RUN=${NR_RUN}  -DNR_EMBEDDING=${NR_EMBEDDING} -DINDEX_PER_BATCH=${INDEX_PER_BATCH} -DNR_ROWS=${NR_ROWS}   -DNR_BATCHES=${NR_BATCHES} -o build/embdpu src/dpu/dpu_embedding.c

clean:
	([ -d "build" ] && rm -r build/ && rm -f out.cpu.log out.dpu.log) || [ ! -d "build" ]

testdpu: FORCE
	./build/emb

tracedpu: FORCE
	dpu-profiling functions -o dpu.json -a -A -f synthetic_inference  -- ./build/emb

FORCE: ;