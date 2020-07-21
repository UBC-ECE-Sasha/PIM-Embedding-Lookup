// to compile the code: gcc -O0 -g3 --std=c99 -o host_multi host_multi.c -g `dpu-pkg-config --cflags --libs dpu`
// to build a shared library: gcc -shared -Wl,-soname,host_multi -o emblib.so -fPIC host_multi.c `dpu-pkg-config --cflags --libs dpu`
#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <dpu_log.h>
#include <stdlib.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./dpu_multi"
#endif

struct dpu_set_t set, dpu;

/*
    Params:
    1. nr_rows: number of rows of the embedding table
    2. nr_cols: number of columns of the embedding table
    3. data: a pointer of the size nr_rows*nr_cols that contains the elements inside the embedding table
    Result:
    This function writes nr_rows, nr_cols, and data using dpu_copy_to from DRAM to MRAM
*/
void populate_mram(uint64_t nr_rows, uint64_t nr_cols, double *data) {

    uint64_t write_len=nr_cols*nr_rows*sizeof(uint64_t);

    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint8_t *)&nr_rows, sizeof(nr_rows)));
    DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint8_t *)&nr_cols, sizeof(nr_cols)));
    
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, 0, (const uint8_t *)data, write_len));

    return;
}

/*
    Params:
    1. ans: a pointer that be updated with the rows that we lookup
    2. input: a pointer containing the specific rows we want to lookup
    3. index_len: contains the number of rows that we want to lookup from the table
    4. nr_rows: number of rows of the embedding table
    5. nr_cols: number of columns of the embedding table
    Result:
    This function updates ans with the elements of the rows that we have lookedup
*/
void lookup(double *ans, uint64_t* input, uint64_t index_len, uint64_t nr_rows, uint64_t nr_cols){

    uint64_t offset=(nr_rows*nr_cols+index_len)*sizeof(uint64_t);

    uint64_t write_len=index_len*sizeof(uint64_t);

    uint64_t read_len=index_len*nr_cols*sizeof(uint64_t);

    DPU_ASSERT(dpu_copy_to(set, "index_len_input", 0, (const uint8_t *)&index_len, sizeof(index_len)));
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, nr_cols*nr_rows*sizeof(uint64_t), (const uint8_t *)input, write_len));

    dpu_launch(set, DPU_SYNCHRONOUS);

    DPU_FOREACH(set, dpu) {
        //DPU_ASSERT(dpu_log_read(dpu, stdout));
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset , (double*)ans, read_len));
        
        for (int i=0; i<index_len; i++){
            for (int j=0; j<nr_cols; j++)
	            printf("ans[%d][%d] = %f\n", (uint64_t)input[i], j, ans[i*nr_cols+j]);
	    }
    }

    dpu_free(set);
}

int main(){
}