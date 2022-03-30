// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common.h"
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

int32_t* buffer_data[NR_COLS];
bool first_run=true;
struct dpu_set_t *dpu_set;

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
    RT_LAUNCH = 1
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

static int alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows) {
    
    for(int j=0; j<NR_COLS; j++){

        size_t sz = nr_rows*sizeof(int32_t);
        buffer_data[j] = (int32_t*)malloc(ALIGN(sz,8));
        if (buffer_data[j] == NULL) {
            return ENOMEM;
        }

        for(int k=0; k<nr_rows; k++){
            buffer_data[j][k] = table_data[k*NR_COLS+j];
        }

    }
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

struct dpu_set_t* populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data, dpu_runtime_totals *runtime){
    struct timespec start, end;

    if(table_id>=AVAILABLE_RANKS){
        fprintf(stderr,"%d ranks available but tried to load table %dth",AVAILABLE_RANKS,table_id);
        exit(1);
    }

    //TIME_NOW(&start);
    if (alloc_buffers(table_id, table_data, nr_rows) != 0) {
        enomem();
    }
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_prepare += TIME_DIFFERENCE(start, end);

    //TIME_NOW(&start);

    struct dpu_set_t dpu, dpu_rank;
    if(first_run){
        dpu_set=malloc(sizeof(struct dpu_set_t));
        DPU_ASSERT(dpu_alloc(NR_COLS*NR_TABLES, NULL, dpu_set));
        DPU_ASSERT(dpu_load(*dpu_set, DPU_BINARY, NULL));
        first_run=false;
    }
    printf("dpu_set ptr in C populate:%p\n",dpu_set);

    uint32_t len;
    uint8_t dpu_id,rank_id;
    
    DPU_RANK_FOREACH(*dpu_set, dpu_rank, rank_id){
        if(rank_id==table_id){
            DPU_FOREACH(dpu_rank, dpu, dpu_id){
                DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_rank,DPU_XFER_TO_DPU, "emb_data", 0, ALIGN(nr_rows*sizeof(int32_t),8), DPU_XFER_DEFAULT));
        }
    }


    for (int i = 0; i < NR_COLS; i++){
        free(buffer_data[i]);
    }
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_populate_copy_in += TIME_DIFFERENCE(start, end);

    return dpu_set;
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
int32_t* lookup(uint32_t** indices, uint32_t** offsets, uint32_t* indices_len,
                uint32_t* nr_batches, float **final_results, struct dpu_set_t *dpu_set_ptr
                //,dpu_runtime_group *runtime_group
                ){
    printf("dpu_set ptr in C lookup:%p\n",dpu_set_ptr);
    //struct timespec start, end;
    int dpu_id,rank_id;
    struct dpu_set_t dpu_rank,dpu;
    struct query_len lengths[NR_TABLES];

    //if (runtime_group && RT_CONFIG == RT_ALL) TIME_NOW(&start);

    DPU_RANK_FOREACH(*dpu_set_ptr,dpu_rank,rank_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu_rank,indices[rank_id]));
        printf("%d\n",rank_id);
    }
    sleep(10);
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
        indices_len[0]*sizeof(uint32_t),8),DPU_XFER_DEFAULT));

    DPU_RANK_FOREACH(*dpu_set_ptr,dpu_rank,rank_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu_rank,offsets[rank_id]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_offsets",0,ALIGN(
        nr_batches[0]*sizeof(uint32_t),8),DPU_XFER_DEFAULT));

    DPU_RANK_FOREACH(*dpu_set_ptr,dpu_rank,rank_id){
        lengths[rank_id].indices_len=*indices_len;
        lengths[rank_id].nr_batches=*nr_batches;
        DPU_ASSERT(dpu_prepare_xfer(dpu_rank,&lengths[rank_id]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_lengths",0,
    sizeof(struct query_len),DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(*dpu_set_ptr, DPU_ASYNCHRONOUS));

    bool done[NR_TABLES], fault[NR_TABLES];
    int32_t tmp_results[NR_COLS][nr_batches[0]];
    DPU_RANK_FOREACH(*dpu_set_ptr,dpu_rank,rank_id){
        dpu_status(dpu_rank, &done[rank_id], &fault[rank_id]);
        if(fault[rank_id]){
            fprintf(stderr,"%dth rank failed launching.\n",rank_id);
            exit(1);
        }
        if(done[rank_id]){
            DPU_FOREACH(dpu_rank, dpu, dpu_id){
                DPU_ASSERT(dpu_prepare_xfer(dpu,&tmp_results[dpu_id][0]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_FROM_DPU, "results",0,
            ALIGN(sizeof(int32_t)*nr_batches[0],8), DPU_XFER_DEFAULT));

            for (int j=0; j<NR_COLS; j++){
                for(int k=0; k<nr_batches[0]; k++)
                    final_results[rank_id][k*NR_COLS+j]=(float)tmp_results[j][k]/pow(10,9);
            }
        }
    }     
    /* if (runtime_group && RT_CONFIG == RT_LAUNCH) {
        if(runtime_group[table_id].in_use >= runtime_group[table_id].length) {
            TIME_NOW(&end);
            fprintf(stderr,
                "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                dpu_id, runtime_group[table_id].in_use, table_id, runtime_group[table_id].length);
            exit(1);
        }
        copy_interval(
            &runtime_group->intervals[runtime_group[table_id].in_use], &start, &end);
            runtime_group[table_id].in_use++;
    } */
    return 0;
}
int
main() {
}