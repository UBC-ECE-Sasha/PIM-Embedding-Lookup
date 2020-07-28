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
#define DPU_BINARY "toy_dpu"
#endif

#define MAX_CAPACITY 2 //MEGABYTE(60)

struct dpu_set_t set, dpu, dpu_rank;

/*
    Params:
    1. nr_rows: array of number of rows of the embedding tables
    2. nr_cols: aaray of number of columns of the embedding tables
    3. data: a pointer of the size |for (embedding in embedding_tables): sum+=nr_rows*nr_cols| 
    that contains the elements inside all the embedding tables

    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk to one dpu as well as number of rows and columns of the corresponding
    table with the first and last index held in each dpu.
*/

void populate_mram(uint32_t nr_emb, uint32_t *nr_rows, uint32_t *nr_cols, int32_t *emb_data){
    uint32_t curr_emb_size=0;
    uint32_t nr_buffer=0;
    uint32_t nr_dpus[nr_emb], indices_len=nr_emb;
    uint32_t *first_indices=(uint32_t*)malloc(nr_emb*sizeof(uint32_t));
    uint32_t *last_indices=(uint32_t*)malloc(nr_emb*sizeof(uint32_t));
    uint32_t curr_nr_rows, curr_nr_cols;

    for (int i=0; i<nr_emb; i++){
        curr_nr_rows=nr_rows[i];
        curr_nr_cols=nr_cols[i];
        curr_emb_size=curr_nr_rows*curr_nr_cols;
        if( curr_emb_size<= MAX_CAPACITY){
            nr_dpus[i]=1;
            first_indices[nr_buffer]=0;
            last_indices[nr_buffer]=curr_emb_size;
            nr_buffer++;
        }
        else{
            nr_dpus[i]=(int)(curr_emb_size/MAX_CAPACITY);
            if(curr_emb_size%MAX_CAPACITY!=0){
                nr_dpus[i]++;
            }
            for(int j=0; j<nr_dpus[i]; j++){
                first_indices[nr_buffer]=j*MAX_CAPACITY;
                last_indices[nr_buffer]=MIN((j+1)*MAX_CAPACITY-1, curr_emb_size);
                nr_buffer++;
                indices_len++;
                first_indices=(uint32_t*)realloc(first_indices, indices_len*sizeof(uint32_t));
                last_indices=(uint32_t*)realloc(last_indices, indices_len*sizeof(uint32_t));
            }
        }
    }

    int32_t emb_buffer[nr_buffer][MAX_CAPACITY];
    uint32_t buffer_ptr=0,data_ptr=0;

    for(int i=0; i<nr_emb; i++){
        for(int j=0; j<nr_dpus[i]; j++){
            for(int t=0, k=first_indices[buffer_ptr]; k<=last_indices[buffer_ptr]; k++, t++){
                emb_buffer[buffer_ptr][t]=emb_data[data_ptr+k];
                //printf("here for %d, %d, %d\n",i,j,emb_data[data_ptr+k]);
            }
            buffer_ptr++;
        }
        data_ptr+=nr_rows[i]*nr_cols[i];
    }

    DPU_ASSERT(dpu_alloc(nr_buffer, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    uint32_t emb_ptr=0;
    uint32_t alloc_buffers=0;
    uint32_t curr_first_index, curr_last_index;
    uint8_t dpu_id,rank_id;
    DPU_FOREACH(set, dpu, dpu_id){
        if(nr_dpus[emb_ptr]==0)
            emb_ptr++;
        curr_nr_cols=nr_cols[emb_ptr];
        curr_nr_rows=nr_rows[emb_ptr];
        curr_first_index=first_indices[dpu_id];
        curr_last_index=last_indices[dpu_id];

        DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint64_t *)&curr_nr_rows, 8));
        DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint64_t *)&curr_nr_cols, 8));
        DPU_ASSERT(dpu_copy_to(set, "first_index_input", 0, (const uint64_t *)&nr_rows, 8));
        DPU_ASSERT(dpu_copy_to(set, "last_index_input", 0, (const uint64_t *)&nr_cols, 8));

        DPU_ASSERT(dpu_prepare_xfer(dpu, emb_buffer[alloc_buffers]));
        //printf("i is:%d, input:%d,%d\n",dpu_id,emb_buffer[alloc_buffers][0],emb_buffer[alloc_buffers][1]);

        nr_dpus[emb_ptr]--;
        alloc_buffers++;
    }
    DPU_RANK_FOREACH(set,dpu_rank, rank_id){
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "data", 0, MAX_CAPACITY*sizeof(int32_t), DPU_XFER_DEFAULT));
    }
    printf("done!\n");

    /* int32_t ans[MAX_CAPACITY];
    DPU_FOREACH(set, dpu, dpu_id){
        dpu_launch(dpu, DPU_SYNCHRONOUS);
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0 , (int32_t*)ans, 2*sizeof(int32_t)));
        printf("%d: %d, %d\n",dpu_id,ans[0], ans[1]);
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    } */

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
    uint32_t row[]={2,2,2,2};
    uint32_t cols[]={4,4,4,4};
    int32_t data[]={1,2,3,4,5,6,7,8,2,4,6,8,10,12,14,16,3,6,9,12,15,18,21,24,4,8,12,16,20,24,28,32};
    populate_mram(4,row,cols, data);
}
