// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common/include/common.h"
#include "host/include/host.h"
#include "emb_types.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define RT_CONFIG 0

#ifndef DPU_BINARY
#    define DPU_BINARY "../upmem/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

uint32_t total_buffers=0, buff_arr_len=NR_TABLES;
uint32_t ready_to_alloc_buffs=0, done_dpus=0, allocated_ranks=0;
struct embedding_buffer *buffers[MAX_NR_BUFFERS];
struct embedding_table *tables[NR_TABLES];
struct dpu_set_t dpu_ranks[AVAILABLE_RANKS];

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct dpu_runtime
 * @brief DPU execution times
 */
typedef struct dpu_runtime_totals {
    double execution_time_prepare;
    double execution_time_populate_copy_in;
    double execution_time_copy_in;
    double execution_time_copy_out;
    double execution_time_aggregate_result;
    double execution_time_launch;
} dpu_runtime_totals;

/**
 * @struct dpu_timespec
 * @brief ....
 */
typedef struct dpu_timespec {
    long tv_nsec;
    long tv_sec;
} dpu_timespec;

/**
 * @struct dpu_runtime_interval
 * @brief DPU execution interval
 */
typedef struct dpu_runtime_interval {
    dpu_timespec start;
    dpu_timespec stop;
} dpu_runtime_interval;

/**
 * @struct dpu_runtime_config
 * @brief ...
 */
typedef enum dpu_runtime_config {
    RT_ALL = 0,
    RT_LAUNCH = 1,
    RT_COPY_TO = 2
} dpu_runtime_config;

/**
 * @struct dpu_runtime_group
 * @brief ...
 */
typedef struct dpu_runtime_group {
    unsigned int in_use;
    unsigned int length;
    dpu_runtime_interval *intervals;
} dpu_runtime_group;

static void enomem() {
    fprintf(stderr, "Out of memory\n");
    exit(ENOMEM);
}

static void copy_interval(dpu_runtime_interval *interval,
                          struct timespec * const start,
                          struct timespec * const end) {
    interval->start.tv_nsec = start->tv_nsec;
    interval->start.tv_sec = start->tv_sec;
    interval->stop.tv_nsec = end->tv_nsec;
    interval->stop.tv_sec = end->tv_sec;
}

static int alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows, uint32_t *first_row, uint32_t *last_row) {
    uint64_t table_size = nr_rows*NR_COLS;

    tables[table_id]=malloc(sizeof(struct embedding_table));

    if (tables[table_id] == NULL) {
        return ENOMEM;
    }

    tables[table_id]->nr_rows = nr_rows;

    tables[table_id]->nr_buffers = (int)((table_size*1.0)/(MAX_CAPACITY*1.0));
    if((table_size%MAX_CAPACITY) != 0){
        tables[table_id]->nr_buffers+=1;
    }
    tables[table_id]->first_dpu_id=total_buffers;
    for(int j=0; j<tables[table_id]->nr_buffers; j++){
        *first_row = j*MAX_CAPACITY/NR_COLS;
        *last_row = MIN(nr_rows-1, ((j+1)*MAX_CAPACITY)/NR_COLS-1);

        buffers[total_buffers] = malloc(sizeof(struct embedding_buffer));
        if (buffers[total_buffers] == NULL) {
            return ENOMEM;
        }

        buffers[total_buffers]->first_row = *first_row;
        buffers[total_buffers]->last_row = *last_row;
        buffers[total_buffers]->table_id = table_id;

        size_t sz = (*last_row-*first_row+1)*NR_COLS*sizeof(int32_t);
        buffers[total_buffers]->data = malloc(ALIGN(sz,8));
        if (buffers[total_buffers]->data == NULL) {
            return ENOMEM;
        }

        for(int k=0; k<(*last_row-*first_row+1)*NR_COLS; k++){
            buffers[total_buffers]->data[k] = table_data[(*first_row*NR_COLS)+k];
        }
        total_buffers++;
    }
    ready_to_alloc_buffs += tables[table_id]->nr_buffers;
    tables[table_id]->last_dpu_id = total_buffers;

    return 0;
}

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

void populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data, dpu_runtime_totals *runtime){
    struct timespec start, end;
    uint32_t first_row, last_row;

    TIME_NOW(&start);
    if (alloc_buffers(table_id, table_data, nr_rows, &first_row, &last_row) != 0) {
        enomem();
    }
    TIME_NOW(&end);

    if (runtime) runtime->execution_time_prepare += TIME_DIFFERENCE(start, end);

    TIME_NOW(&start);

    // Done with analyzing all tables or nr ready_to_alloc_buffs enough for a rank so
    // allocate a rank and copy embedding data.
    if (ready_to_alloc_buffs >= DPUS_PER_RANK || table_id == NR_TABLES - 1) {
        struct dpu_set_t set, dpu, dpu_rank;
        //printf("allocating %d dpus and %d dpus allocated before.\n", ready_to_alloc_buffs, done_dpus);
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
            DPU_ASSERT(dpu_prepare_xfer(dpu, buffers[done_dpus+dpu_id]));
        }
        DPU_ASSERT(dpu_push_xfer(set,DPU_XFER_TO_DPU, "emb_buffer", 0, sizeof(struct embedding_buffer), DPU_XFER_DEFAULT));


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
    TIME_NOW(&end);

    if (runtime) runtime->execution_time_populate_copy_in += TIME_DIFFERENCE(start, end);

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
int32_t* lookup(uint32_t* indices, uint32_t *offsets, uint64_t *indices_len,
                uint64_t *offsets_len, int32_t *final_results,
                dpu_runtime_group *runtime_group){
    struct timespec start, end;
    int dpu_id, tmp_ptr=0, table_ptr=0, indices_ptr=0, offsets_ptr=0, max_len=0;
    uint64_t copied_indices;
    struct dpu_set_t dpu;

    if (runtime_group && (RT_CONFIG == RT_ALL || RT_CONFIG == RT_COPY_TO)) {
        dbg_printf("%s", "START - RT_CONFIG == RT_ALL | RT_COPY_TO\n");
        TIME_NOW(&start);
    }

    for(int k=0; k<allocated_ranks; k++){
        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){
            if(tables[table_ptr]->nr_buffers==tmp_ptr){
                if(indices_len[table_ptr]>max_len)
                    max_len=indices_len[table_ptr];
                tmp_ptr=0;
                indices_ptr+=indices_len[table_ptr];
                offsets_ptr+=offsets_len[table_ptr];
                table_ptr++;
            }
            copied_indices=0;
            while(copied_indices<indices_len[table_ptr]){
                DPU_ASSERT(dpu_copy_to(dpu, "input_indices" , copied_indices*sizeof(uint32_t), (const uint32_t *)&indices[indices_ptr+copied_indices],
                ALIGN(MIN(2048,(indices_len[table_ptr]-copied_indices)*sizeof(uint32_t)),8)));
                copied_indices+=2048/sizeof(uint32_t);
            }
            DPU_ASSERT(dpu_copy_to(dpu, "input_offsets" , 0, (const uint32_t *)&offsets[offsets_ptr], ALIGN(offsets_len[table_ptr]*sizeof(uint32_t),8)));
            DPU_ASSERT(dpu_copy_to(dpu, "input_nr_indices" , 0, &indices_len[table_ptr], sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "input_nr_offsets" , 0, &offsets_len[table_ptr], sizeof(uint64_t)));
            tmp_ptr++;
        }
    }

    if (runtime_group && RT_CONFIG == RT_COPY_TO) {
        dbg_printf("%s", "STOP - RT_CONFIG == RT_COPY_TO\n");
        TIME_NOW(&end);
    }

    // run dpus
    for( int k=0; k<allocated_ranks; k++){

        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){

            if (runtime_group && RT_CONFIG == RT_LAUNCH) {
                dbg_printf("%s", "STOP - RT_CONFIG == RT_LAUNCH\n");
                TIME_NOW(&start);
            }

            uint64_t tmp_int=1;
            DPU_ASSERT(dpu_copy_to(dpu, "first_run" , 0, &tmp_int, sizeof(uint64_t)));
            DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));

            if (runtime_group && RT_CONFIG == RT_LAUNCH) {
                if(runtime_group[dpu_id].in_use >= runtime_group[dpu_id].length) {
                    dbg_printf("%s", "STOP - RT_CONFIG == RT_LAUNCH\n");
                    TIME_NOW(&end);
                    fprintf(stderr,
                        "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                        dpu_id, runtime_group[dpu_id].in_use, dpu_id, runtime_group[dpu_id].length);
                    exit(1);
                }
                copy_interval(
                    &runtime_group->intervals[runtime_group[dpu_id].in_use], &start, &end);
                    runtime_group[dpu_id].in_use++;
            }
        }
    }

    uint64_t nr_batches;
    struct lookup_result *partial_results[done_dpus];
    for( int k=0; k<allocated_ranks; k++){
        DPU_FOREACH(dpu_ranks[k], dpu, dpu_id){
            DPU_ASSERT(dpu_copy_from(dpu, "input_nr_offsets", 0 , &nr_batches, sizeof(uint64_t)));
            partial_results[dpu_id]=malloc(sizeof(struct lookup_result)*nr_batches);
            DPU_ASSERT(dpu_copy_from(dpu, "results", 0, &partial_results[dpu_id][0], ALIGN(sizeof(struct lookup_result)*nr_batches,8)));

            if (runtime_group) {
                // for (int i = 0; i < NR_DPUS; i++) {
                //     printf("runtime_group[%d].in_use = %d, runtime_group[%d].length = %d\n",
                //     i, runtime_group[i].in_use, i, runtime_group[i].in_use);
                // }
                if (runtime_group && RT_CONFIG == RT_ALL) {
                    dbg_printf("%s", "STOP - RT_CONFIG == RT_ALL\n");
                    TIME_NOW(&end);
                }
                if(runtime_group[dpu_id].in_use >= runtime_group[dpu_id].length) {
                    fprintf(stderr,
                        "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                        dpu_id, runtime_group[dpu_id].in_use, dpu_id, runtime_group[dpu_id].length);
                    exit(1);
                }
                copy_interval(
                        &runtime_group->intervals[runtime_group[dpu_id].in_use], &start, &end);
                runtime_group[dpu_id].in_use++;
            }
        }
    }

    int result_ptr=0, data_ptr=0;
    int32_t tmp_result[NR_COLS];
    for( int k=0; k<NR_TABLES; k++){
        if(tables[k]->nr_buffers==1){
            for( int j=0; j<offsets_len[k]; j++){
                for(int i=0; i<NR_COLS; i++){
                    final_results[data_ptr+i]=partial_results[result_ptr][j].data[i];
                    //printf("final_result[%d]=%d\n",data_ptr+i,final_results[data_ptr+i]);
                }
                data_ptr+=NR_COLS;
            }
            result_ptr++;
        }
        else{
            for( int j=0; j<offsets_len[k]; j++){
                for (int l=0; l<NR_COLS; l++)
                    tmp_result[l]=0;
                for( int t=0; t< tables[k]->nr_buffers; t++){
                    for(int i=0; i<NR_COLS; i++){
                        tmp_result[i]+=partial_results[result_ptr+t][j].data[i];
                    }
                }
                for( int i=0; i< NR_COLS; i++){
                    final_results[data_ptr+i]=tmp_result[i];
                }
                data_ptr+=NR_COLS;
            }
            result_ptr+=tables[k]->nr_buffers;
        }
    }
    return 0;
}
int
main() {
}
