// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags --libs dpu`
// to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <dpu_log.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_"
#endif

#define MAX_CAPACITY MEGABYTE(10) //Must be a multiply of 2
#define NR_TABLES 26

struct dpu_set_t set, dpu, dpu_rank;
uint32_t nr_buffer=0, nr_dpus[NR_TABLES], indices_len=NR_TABLES;
uint64_t *first_indices;
uint64_t *last_indices;
uint32_t allocated_dpus=0, buffer_ptr=0;

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

void populate_mram(uint32_t table_id, uint32_t nr_rows, uint32_t nr_cols, int32_t *emb_data){
    uint64_t table_size=nr_rows*nr_cols;
    uint32_t max_buffer_len;

    if (table_id==0){
        first_indices=(uint64_t*)malloc(NR_TABLES*sizeof(uint64_t));
        last_indices=(uint64_t*)malloc(NR_TABLES*sizeof(uint64_t));
    }

    if( table_size <= MAX_CAPACITY){
        nr_dpus[table_id]=1;
        first_indices[nr_buffer]=0;
        last_indices[nr_buffer]=table_size-1;
        nr_buffer++;
        max_buffer_len=table_size;
    }
    else{
        nr_dpus[table_id]=(int)((table_size*1.0)/(MAX_CAPACITY*1.0));
        if(table_size%MAX_CAPACITY!=0){
            nr_dpus[table_id]++;
        }
        for(int j=0; j<nr_dpus[table_id]; j++){
            first_indices[nr_buffer]=j*MAX_CAPACITY;
            //last_indices[nr_buffer]=MIN( ((j+1)*(MAX_CAPACITY))-1 , table_size-1);
            if(table_size<((j+1)*(MAX_CAPACITY)-1))
                last_indices[nr_buffer]=table_size-1;
            else
                last_indices[nr_buffer]=(j+1)*(MAX_CAPACITY)-1;
            nr_buffer++;
            indices_len++;
            first_indices=(uint64_t*)realloc(first_indices, indices_len*sizeof(uint64_t));
            last_indices=(uint64_t*)realloc(last_indices, indices_len*sizeof(uint64_t));
            max_buffer_len=MAX_CAPACITY;
        }
    }
    printf("first traverse of %dth table of size %d with %d buffers of %d each and %d buffers up to now.\n",
    table_id,table_size,nr_dpus[table_id], MAX_CAPACITY, nr_buffer);

    int32_t *emb_buffer[nr_dpus[table_id]];
    uint32_t first_index, last_index;

    for(int j=0; j<nr_dpus[table_id]; j++){
        first_index=first_indices[buffer_ptr];
        last_index=last_indices[buffer_ptr];
        printf("mallocing %d for %dth buffer.\n",last_index-first_index+1,buffer_ptr);
        emb_buffer[j]=(int32_t*)malloc((last_index-first_index+1)*sizeof(int32_t));
        for(int k=0; k<last_index-first_index+1; k++){
            emb_buffer[j][k]=emb_data[first_index+k];
            //printf("here for %d element of %d buffer from %d data\n",k,j,first_index*j+k);
        }
        buffer_ptr++;
    }

    printf("allocating %d dpus and %d dpus allocated before.\n",nr_dpus[table_id],allocated_dpus);
    printf("%s",dpu_alloc(nr_dpus[table_id], NULL, &set));
    printf("DPUs allocated\n");

    uint32_t len;
    uint8_t dpu_id,rank_id;
    DPU_FOREACH(set, dpu, dpu_id){
        first_index=first_indices[allocated_dpus+dpu_id];
        last_index=last_indices[allocated_dpus+dpu_id];
        len= last_index-first_index+1;

        if(len<len< (MEGABYTE(1)/16))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_64KB", NULL));
        else if(len< (MEGABYTE(1)/8))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_125KB", NULL));
        else if( len< (MEGABYTE(1)/4) )
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_250KB", NULL));
        else if (len< (MEGABYTE(1)/2))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_250KB", NULL));
        else if(len< MEGABYTE(1))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_1MB", NULL));
        else if(len< MEGABYTE(2))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_2MB", NULL));
        else if (len< MEGABYTE(4))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_4MB", NULL));
        else if (len<MEGABYTE(8))
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_8MB", NULL));
        else
            DPU_ASSERT(dpu_load(dpu, "/home/upmem0016/niloo/PIM-Embedding-Lookup/upmem/toy_dpu_10MB", NULL));

        DPU_ASSERT(dpu_copy_to(dpu, "row_size_input", 0, (const uint64_t *)&nr_rows, sizeof(uint64_t)));
        DPU_ASSERT(dpu_copy_to(dpu, "col_size_input", 0, (const uint64_t *)&nr_cols, sizeof(uint64_t)));
        DPU_ASSERT(dpu_copy_to(dpu, "first_index_input", 0, (const uint64_t *)&first_index, sizeof(uint64_t)));
        DPU_ASSERT(dpu_copy_to(dpu, "last_index_input", 0, (const uint64_t *)&last_index, sizeof(uint64_t)));

        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, (const int32_t *)emb_buffer[dpu_id], ALIGN(len,8)));
        //DPU_ASSERT(dpu_prepare_xfer(dpu, emb_buffer[alloc_buffers]));
        //printf("i is:%d, input:%d,%d\n",dpu_id,emb_buffer[alloc_buffers][0],emb_buffer[alloc_buffers][1]);
    }
    printf("buffers prepared\n");
     allocated_dpus+=nr_dpus[table_id];

    /* DPU_RANK_FOREACH(set,dpu_rank, rank_id){
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "data", 0, max_buffer_len*sizeof(int32_t), DPU_XFER_DEFAULT));
    } */
    printf("done with table %d!\n",table_id);

    for (int i=0; i<nr_dpus[table_id]; i++)
        free(emb_buffer[i]);

    /* int32_t ans[MAX_CAPACITY];
    DPU_FOREACH(set, dpu, dpu_id){
        dpu_launch(dpu, DPU_SYNCHRONOUS);
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0 , (int32_t*)ans, 2*sizeof(int32_t)));
        printf("%d: %d, %d\n",dpu_id,ans[0], ans[1]);
        //DPU_ASSERT(dpu_log_read(dpu, stdout));
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
    //uint32_t row[]={2,3,2,2,2};
    //uint32_t cols[]={4,3,4,4,1};
    //int32_t data[]={1,2,3,4,5,6,7,8,2,4,6,8,10,12,14,16,18,3,6,9,12,15,18,21,24,4,8,12,16,20,24,28,32,5,10};
    //populate_mram(5,row,cols, data);
}
