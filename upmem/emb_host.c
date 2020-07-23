// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags --libs dpu`
// to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <dpu_log.h>
#include <stdlib.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./emb_dpu_lookup"
#endif

#define MAX_CAPACITY MEGABYTE(60)

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



void copy_emb(uint32_t nr_emb, uint32_t *nr_rows, uint32_t *nr_cols, int32_t *emb_data){
    uint32_t curr_emb_size=0;
    int32_t **emb_buffer=(int32_t**)malloc(26*sizeof(int32_t*));
    uint32_t data_ptr=0;
    uint32_t nr_buffer=0;
    uint32_t *nr_dpus, *first_indices, *last_indices;
    uint32_t curr_nr_rows, curr_nr_cols;
    struct dpu_set_t set, dpu, dpu_rank;

    for (int i=0; i<nr_emb; i++){
        curr_nr_rows=nr_rows[i];
        curr_nr_cols=nr_cols[i];
        curr_emb_size=curr_nr_rows*curr_nr_cols;
        if(curr_emb_size<MAX_CAPACITY){
            nr_dpus[i]=1;
            first_indices[nr_buffer]=0;
            last_indices[nr_buffer]=curr_emb_size;
            emb_buffer[nr_buffer]=(int32_t*)malloc(curr_emb_size*sizeof(int32_t));
            for (int j=0; j<curr_emb_size; j++){
                emb_buffer[nr_buffer][j]=emb_data[data_ptr+j];
            }
            nr_buffer++;
            data_ptr+=curr_emb_size;
        }
        else{
            nr_dpus[i]=0;
            while(curr_emb_size>MAX_CAPACITY){
                nr_dpus[i]++;
                first_indices[nr_buffer]=curr_nr_cols*curr_nr_rows-curr_emb_size;
                last_indices[nr_buffer]=first_indices[nr_buffer]+MAX_CAPACITY;
                emb_buffer[nr_buffer]=(int32_t*)malloc(MAX_CAPACITY*sizeof(int32_t));
                for (int j=0; j<MAX_CAPACITY; j++){
                emb_buffer[nr_buffer][j]=emb_data[data_ptr+j];
                }
                nr_buffer++;
                emb_buffer=(int32_t**)realloc(emb_buffer, nr_buffer*sizeof(int32_t*));
                curr_emb_size-=MAX_CAPACITY;
                data_ptr+=MAX_CAPACITY;
            }
            if(curr_emb_size>0){
                first_indices[nr_buffer]=curr_nr_cols*curr_nr_rows-curr_emb_size;
                last_indices[nr_buffer]=curr_nr_cols*curr_nr_rows;
                emb_buffer[nr_buffer]=(int32_t*)malloc(curr_emb_size*sizeof(int32_t));
                for (int j=0; j<curr_emb_size; j++){
                    emb_buffer[nr_buffer][j]=emb_data[data_ptr+j];
                }
                nr_buffer++;
                nr_dpus[i]++;
                data_ptr+=curr_emb_size;
            }
        }
    }
    DPU_ASSERT(dpu_alloc(nr_buffer, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    uint32_t emb_ptr=0;
    uint32_t alloc_buffers=0;
    uint32_t curr_first_index, curr_last_index;
    uint8_t dpu_id,rank_id;
    DPU_FOREACH(set, dpu, dpu_id){
        if(nr_dpus[emb_ptr]>0){
            curr_nr_cols=nr_cols[emb_ptr];
            curr_nr_rows=nr_rows[emb_ptr];
            curr_first_index=first_indices[dpu_id];
            curr_last_index=last_indices[dpu_id];

            DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint32_t *)&curr_nr_rows, sizeof(curr_nr_rows)));
            DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint32_t *)&curr_nr_cols, sizeof(curr_nr_cols)));
            DPU_ASSERT(dpu_copy_to(set, "first_index_input", 0, (const uint32_t *)&nr_rows, sizeof(curr_first_index)));
            DPU_ASSERT(dpu_copy_to(set, "last_index_input", 0, (const uint32_t *)&nr_cols, sizeof(curr_last_index)));

            DPU_ASSERT(dpu_prepare_xfer(dpu, emb_buffer[alloc_buffers]));
            free(emb_buffer[alloc_buffers]);

            nr_dpus[emb_ptr]--;
            alloc_buffers++;
        }
        else
            emb_ptr++;
    }
    DPU_RANK_FOREACH(set,dpu_rank, rank_id){
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "emb_input", 0, MAX_CAPACITY, DPU_XFER_DEFAULT));
    }
    return;
}



/*
    Params:
    1. ans: a pointer that be updated with the rows that we lookup
    2. input: a pointer containing the specific rows we want to lookup
    3. length: contains the number of rows that we want to lookup from the table
    4. nr_rows: number of rows of the embedding table
    5. nr_cols: number of columns of the embedding table

    Result:
    This function updates ans with the elements of the rows that we have lookedup
*/
void lookup(int32_t *ans, int32_t* input, uint64_t length, uint64_t nr_rows, uint64_t nr_cols){

    uint64_t offset=nr_cols*nr_rows*sizeof(uint64_t);
    uint64_t write_len=length*sizeof(uint64_t);
    uint64_t read_len=length*nr_cols*sizeof(uint64_t);

    DPU_ASSERT(dpu_copy_to(set, "index_len_input", 0, (const uint8_t *)&length, sizeof(length)));
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, nr_cols*nr_rows*sizeof(uint64_t), (const uint8_t *)input, write_len));

    dpu_launch(set, DPU_SYNCHRONOUS);

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset , (int32_t*)ans, read_len));
        for (int i=0; i<length; i++){
            for (int j=0; j<nr_cols; j++)
	            printf("ans[%d][%d] = %d\n", (int32_t)input[i], j, ans[i*nr_cols+j]);
	    }
    }
    dpu_free(set);
}

int main(){
    uint32_t row[]={2,3};
    uint32_t cols[]={2,3};
    int32_t data[]={1,2,3,4,2,4,6,8,10,12,14,16,18};
    copy_emb(2,row,cols,data);
}
