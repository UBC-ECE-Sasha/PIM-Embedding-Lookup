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
#define DPU_BINARY "../upmem/toy_dpu" //Relative path regarding the PyTorch code
#endif

#define MAX_CAPACITY MEGABYTE(14) //Must be a multiply of 2
#define NR_TABLES 26
#define DPUS_PER_RANK 64


struct embedding_buffer{
    int32_t *data;
    uint64_t first_index, last_index, first_row, last_row;
    uint64_t nr_rows, nr_cols;
    struct dpu_set_t *dpu;
    uint32_t table_id;
};

struct embedding_table{
    uint32_t first_buffer_index, last_buffer_index, nr_buffers;
    uint64_t nr_rows, nr_cols;
    struct dpu_set_t **dpus;
};

uint32_t total_buffers=0, arrays_len=NR_TABLES;
uint32_t ready_buffers=0, allocated_dpus=0;
struct dpu_set_t *table_dpus[NR_TABLES];
struct embedding_buffer **buffers;
struct embedding_table tables[NR_TABLES];
/*
    Params:
    0. table_id: embedding table number.
    1. nr_rows: number of rows of the embedding table
    2. nr_cols: number of columns of the embedding table
    3. table_data: a pointer of the size nr_rows*nr_cols containing table's data

    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
    corresponding table with the first and last index held in each dpu.
*/

void populate_mram(uint32_t table_id, uint64_t nr_rows, uint64_t nr_cols, int32_t *table_data){
    uint64_t table_size=nr_rows*nr_cols;

    if (table_id==0){
        buffers=(struct embedding_buffer**)malloc(NR_TABLES*sizeof(struct embedding_buffer*));
    }

    if( table_size <= MAX_CAPACITY){
        tables[table_id].nr_buffers=1;
        buffers[total_buffers]=(struct embedding_buffer*)malloc(sizeof(struct embedding_buffer));
        buffers[total_buffers]->first_index=0;
        buffers[total_buffers]->last_index=table_size-1;
        buffers[total_buffers]->first_row=0;
        buffers[total_buffers]->last_row=nr_rows-1;
        buffers[total_buffers]->table_id=table_id;
    }
    else{
        tables[table_id].nr_buffers=(int)((table_size*1.0)/(MAX_CAPACITY*1.0));
        if(table_size%MAX_CAPACITY!=0){
            tables[table_id].nr_buffers+=1;
        }
        for(int j=0; j<tables[table_id].nr_buffers; j++){
            arrays_len++;
            buffers[total_buffers+j]=(struct embedding_buffer*)malloc(sizeof(struct embedding_buffer));
            buffers[total_buffers+j]->first_index=j*MAX_CAPACITY;
            buffers[total_buffers+j]->last_index=MIN(table_size-1, ((j+1)*MAX_CAPACITY)-1);
            buffers[total_buffers+j]->first_row=j*MAX_CAPACITY/nr_cols;
            buffers[total_buffers+j]->last_row=MIN(nr_rows-1, ((j+1)*MAX_CAPACITY)/nr_cols-1);
            buffers=(struct embedding_buffer**)realloc(buffers, arrays_len*(sizeof(struct embedding_buffer*)));
            buffers[total_buffers+j]->table_id=table_id;
        }
    tables[table_id].first_buffer_index=total_buffers;
    tables[table_id].last_buffer_index=total_buffers+tables[table_id].nr_buffers;
    }
    //printf("first traverse of %dth table of size %d with %d buffers of %d each and %d buffers up to now.\n",
    //table_id,table_size,buffer_per_table[table_id], MAX_CAPACITY, total_buffers);

    uint32_t first_index, last_index;

    for(int j=0; j<tables[table_id].nr_buffers; j++){
        first_index=buffers[total_buffers]->first_index;
        last_index=buffers[total_buffers]->last_index;
        //printf("mallocing %d for %dth buffer of %d.\n",ALIGN((last_index-first_index+1)*sizeof(int32_t),8), ready_buffers,total_buffers);
        buffers[total_buffers]->data=(int32_t*)malloc(ALIGN((last_index-first_index+1)*sizeof(int32_t),8));
        for(int k=0; k<last_index-first_index+1; k++){
            buffers[total_buffers]->data[k]=table_data[first_index+k];
        }
        //printf("first: %d, last: %d for %d buffer\n",emb_buffer[ready_buffers][0],emb_buffer[ready_buffers][last_index-first_index],total_buffers);
        ready_buffers++;
        total_buffers++;
    }
    printf("done with %d table\n",table_id);

    if(ready_buffers==DPUS_PER_RANK || table_id==NR_TABLES-1){
        struct dpu_set_t set, dpu, dpu_rank;
        printf("allocating %d dpus and %d dpus allocated before.\n",ready_buffers,allocated_dpus);
        DPU_ASSERT(dpu_alloc(ready_buffers, NULL, &set));
        DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

        uint32_t len;
        uint8_t dpu_id,rank_id;
        DPU_FOREACH(set, dpu, dpu_id){
            first_index=buffers[allocated_dpus+dpu_id]->first_index;
            last_index=buffers[allocated_dpus+dpu_id]->last_index;
            buffers[allocated_dpus+dpu_id]->dpu=&dpu;
            len= last_index-first_index+1;

            DPU_ASSERT(dpu_copy_to(dpu, "row_size_input", 0, (const uint64_t *)&buffers[allocated_dpus+dpu_id]->nr_rows, sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "col_size_input", 0, (const uint64_t *)&buffers[allocated_dpus+dpu_id]->nr_cols, sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "first_index_input", 0, (const uint64_t *)&first_index, sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "last_index_input", 0, (const uint64_t *)&last_index, sizeof(uint64_t)));

            DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, (const int32_t *)buffers[allocated_dpus+dpu_id]->data, ALIGN(len*sizeof(int32_t),8)));
            printf("copied %d buffer to dpu\n",dpu_id);

            buffers[allocated_dpus+dpu_id]->dpu=&dpu;
        }
        /* int32_t ans[4];
        DPU_FOREACH(set, dpu, dpu_id){
            dpu_launch(dpu, DPU_SYNCHRONOUS);
            first_index=buffers[allocated_dpus+dpu_id]->first_index;
            last_index=buffers[allocated_dpus+dpu_id]->last_index;
            printf("first: %d, 2nd: %d, -1: %d, last: %dfor %d buffer\n",buffers[dpu_id]->data[0],buffers[dpu_id]->data[1],
            buffers[dpu_id]->data[last_index-first_index-1],buffers[dpu_id]->data[last_index-first_index],dpu_id);
            uint32_t offset= (last_index-first_index+1)*sizeof(int32_t);
            DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset , (int32_t*)ans, 4*sizeof(int32_t)));
            //printf("%d: %d, %d\n",dpu_id,ans[0], ans[1]);
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }*/

    for (int i=0; i<ready_buffers; i++)
        free(buffers[i]->data);

    for (int i=0; i<NR_TABLES; i++){
        tables[i].dpus=(struct dpu_set_t**)malloc(tables[i].nr_buffers*sizeof(struct dpu_set_t*));
    }
    uint32_t table_ptr=0,tmp_ptr=0;
    DPU_FOREACH(set, dpu,dpu_id){
        if(tables[table_ptr].nr_buffers==tmp_ptr){
            table_ptr++;
            tmp_ptr=0;
        }
        tables[table_ptr].dpus[tmp_ptr]=&dpu;
        tables[table_ptr].first_buffer_index=
        tmp_ptr++;
    }
    allocated_dpus+=ready_buffers;
    ready_buffers=0;
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
    struct dpu_set_t set, dpu, dpu_rank;

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

    //int32_t data[]={1,2,3,4,5,6,7,2,4,6,8,10,12,14,3,6,9,12,15,18,21,4,8,12,16,20,24,28};
    //populate_mram(0,4,7, data);
}
