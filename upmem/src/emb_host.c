// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common/include/common.h"
#include "emb_types.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef DPU_BINARY
#    define DPU_BINARY "../upmem/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

uint32_t total_buffers=0, buff_arr_len=NR_TABLES;
uint32_t ready_to_alloc_buffs=0, done_dpus=0, allocated_ranks=0;
struct embedding_buffer *buffers[MAX_NR_BUFFERS];
struct embedding_table *tables[NR_TABLES];
struct dpu_set_t dpu_ranks[AVAILABLE_RANKS];
/*
    Params:
    0. table_id: embedding table number.
    1. nr_rows: number of rows of the embedding table
    2. NR_COLS: number of columns of the embedding table
    3. table_data: a pointer of the size nr_rows*NR_COLS containing table's data

    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
    corresponding table with the index of the first and last row held in each dpu.
*/

void populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data){
    uint64_t table_size=nr_rows*NR_COLS;
    tables[table_id]=malloc(sizeof(struct embedding_table));
    tables[table_id]->nr_rows=nr_rows;

    tables[table_id]->nr_buffers=(int)((table_size*1.0)/(MAX_CAPACITY*1.0));
    if(table_size%MAX_CAPACITY!=0){
            tables[table_id]->nr_buffers+=1;
        }
    tables[table_id]->first_dpu_id=total_buffers;
    uint32_t first_row, last_row;
    for(int j=0; j<tables[table_id]->nr_buffers; j++){
            buffers[total_buffers]=malloc(sizeof(struct embedding_buffer));
            buffers[total_buffers]->first_row=first_row=j*MAX_CAPACITY/NR_COLS;
            buffers[total_buffers]->last_row=last_row=MIN(nr_rows-1, ((j+1)*MAX_CAPACITY)/NR_COLS-1);
            buffers[total_buffers]->table_id=table_id;
            buffers[total_buffers]->data=malloc(ALIGN((last_row-first_row+1)*NR_COLS*sizeof(int32_t),8));
            for(int k=0; k<(last_row-first_row+1)*NR_COLS; k++){
                buffers[total_buffers]->data[k]=table_data[(first_row*NR_COLS)+k];
            }
            total_buffers++;
        }
    ready_to_alloc_buffs+=tables[table_id]->nr_buffers;
    tables[table_id]->last_dpu_id=total_buffers;

    printf("done with %dth table\n", table_id);

    // Done with analyzing all tables or nr ready_to_alloc_buffs enough for a rank so
    // allocate a rank and copy embedding data.
    if (ready_to_alloc_buffs >= DPUS_PER_RANK || table_id == NR_TABLES - 1) {
        struct dpu_set_t set, dpu, dpu_rank;
        printf("allocating %d dpus and %d dpus allocated before.\n", ready_to_alloc_buffs, done_dpus);
        if (ready_to_alloc_buffs <= DPUS_PER_RANK)
            DPU_ASSERT(dpu_alloc(ready_to_alloc_buffs, NULL, &set));
        else
            DPU_ASSERT(dpu_alloc(DPUS_PER_RANK, NULL, &set));
        DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

        uint32_t len;
        uint8_t dpu_id,rank_id;
        DPU_FOREACH(set, dpu, dpu_id){
            len= (buffers[done_dpus+dpu_id]->last_row-buffers[done_dpus+dpu_id]->first_row+1)*NR_COLS;
            DPU_ASSERT(dpu_copy_to(dpu, "emb_data" , 0, (const int32_t *)buffers[done_dpus+dpu_id]->data, ALIGN(len*sizeof(int32_t),8)));
            DPU_ASSERT(dpu_prepare_xfer(dpu, &buffers[done_dpus+dpu_id]));
            printf("copied %dth buffer to dpu\n",dpu_id);
        }
        DPU_ASSERT(dpu_push_xfer(set,DPU_XFER_TO_DPU, "emb_buffer", 0, sizeof(struct embedding_buffer), DPU_XFER_DEFAULT));

        // This section is just for testing, see if correct values are in DPU
        /* int32_t ans[4];
        DPU_FOREACH(set, dpu, dpu_id){
            dpu_launch(dpu, DPU_SYNCHRONOUS);
            first_row=buffers[done_dpus+dpu_id]->first_row;
            last_row=buffers[done_dpus+dpu_id]->last_row;
            printf("In host last_row:%d, first row:%d & NR_COLS:%d\n",last_row,first_row,NR_COLS); 
            printf("first: %d, 2nd: %d, -1: %d, last: %dfor %d buffer\n",buffers[dpu_id]->data[0],
            buffers[dpu_id]->data[1],
            buffers[dpu_id]->data[(last_row-first_row+1)*NR_COLS-2],
            buffers[dpu_id]->data[(last_row-first_row+1)*NR_COLS-1],dpu_id);
            uint32_t offset=
            ALIGN((last_row-first_row+1)*NR_COLS*sizeof(int32_t),8);
            printf("copying from dpu\n");
            DPU_ASSERT(dpu_copy_from(dpu, "ans_buffer", 0 , (int32_t*)ans, 2*sizeof(int32_t)));
            printf("copied from dpu\n");
            printf("%d: %d, %d\n",dpu_id,ans[0], ans[1]);
            printf("reading log\n");
            DPU_ASSERT(dpu_log_read(dpu, stdout));
            printf("log read for dpu %dth\n",dpu_id);
            printf("------------------------------------------\n");
            //printf("log printed");
        } */

        for (int i = done_dpus; i < ready_to_alloc_buffs; i++)
            free(buffers[i]->data);

        // Assign dpus allocated to buffers to their embedding_tables.
        for (int i=0; i<NR_TABLES; i++){
            tables[i]->buffers=malloc(tables[i]->nr_buffers*sizeof(struct embedding_buffer*));
        }

        uint32_t table_ptr=0,tmp_ptr=0;
        DPU_FOREACH(set, dpu, dpu_id){
            if(tables[table_ptr]->nr_buffers==tmp_ptr){
                table_ptr++;
                tmp_ptr = 0;
            }
            tables[table_ptr]->buffers[tmp_ptr]=buffers[dpu_id];

        }

        // done with a set of dpus, make changes to their counters to move to next set.
        if (ready_to_alloc_buffs <= DPUS_PER_RANK) {
            done_dpus += ready_to_alloc_buffs;
            ready_to_alloc_buffs = 0;
        } else {
            done_dpus += DPUS_PER_RANK;
            ready_to_alloc_buffs -= DPUS_PER_RANK;
        }
        dpu_ranks[allocated_ranks] = set;
        allocated_ranks++;
    }
    return;
}

/*
    Params:
    1. ans: a pointer that be updated with the rows that we lookup
    2. input: a pointer containing the specific rows we want to lookup
    3. length: contains the number of rows that we want to lookup from the table
    4. nr_rows: number of rows of the embedding table
    5. NR_COLS: number of columns of the embedding table

    Result:
    This function updates ans with the elements of the rows that we have lookedup
*/
int32_t* lookup(uint32_t* indices, uint32_t *offsets, uint64_t *indices_len, uint64_t *offsets_len, uint32_t *lookup_ans){
    printf("doing lookup\n");
    int dpu_id, tmp_ptr=0, table_ptr=0, indices_ptr=0, offsets_ptr=0, max_len=0;
    struct dpu_set_t dpu;
    for( int k=0; k<allocated_ranks; k++){
        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){
            if(tables[table_ptr]->nr_buffers==tmp_ptr){
                if(indices_len[table_ptr]>max_len)
                    max_len=indices_len[table_ptr];
                tmp_ptr=0;
                indices_ptr+=indices_len[table_ptr];
                offsets_ptr+=offsets_len[table_ptr];
                table_ptr++;
            }
            printf("indices_len[%d]=%d and offsets_len[%d]=%d\n",table_ptr,indices_len[table_ptr],table_ptr,offsets_len[table_ptr]);
            DPU_ASSERT(dpu_copy_to(dpu, "input_indices" , 0, (const uint32_t *)&indices[indices_ptr], ALIGN(indices_len[table_ptr]*sizeof(uint32_t),8)));
            DPU_ASSERT(dpu_copy_to(dpu, "input_offsets" , 0, (const uint32_t *)&offsets[offsets_ptr], ALIGN(offsets_len[table_ptr]*sizeof(uint32_t),8)));
            DPU_ASSERT(dpu_copy_to(dpu, "input_nr_indices" , 0, &indices_len[table_ptr], sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "input_nr_offsets" , 0, &offsets_len[table_ptr], sizeof(uint64_t)));
            tmp_ptr++;
        }
    }
    printf("done with lookup data copy\n");

    // run dpus
    for( int k=0; k<allocated_ranks; k++){
        printf("doing %d th rank\n",k);
        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){
            printf("launching %d th dpu\n",dpu_id);
            DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
    }
    printf("DPUs done launching\n");
    
     uint64_t nr_batches;
    struct lookup_result *partial_results[done_dpus];
    for( int k=0; k<allocated_ranks; k++){
        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){
            DPU_ASSERT(dpu_copy_from(dpu, "input_nr_offsets", 0 , &nr_batches, sizeof(uint64_t)));
            partial_results[dpu_id]=malloc(sizeof(struct lookup_result)*nr_batches);
            DPU_ASSERT(dpu_copy_from(dpu, "results", 0, partial_results[dpu_id], ALIGN(sizeof(struct lookup_result)*nr_batches,8)));
            printf("%d dpu result:\n",dpu_id);
            for( int j=0; j<nr_batches; j++){
                printf("%d batch:\n", j);
                for (int i=0;i<NR_COLS; i++)
                    printf("%d,",partial_results[dpu_id][j].data[i]);
                printf("\n");
            }
            printf("\n------------------\n");
                
        }
    }
    printf("Done with copying back results\n");

    return 0;
}

int
main() {
}
