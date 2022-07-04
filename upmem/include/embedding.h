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
 * @struct dpu_runtime_group
 * @brief ...
 */
typedef struct dpu_runtime_group {
    unsigned int in_use;
    unsigned int length;
    dpu_runtime_interval *intervals;
} dpu_runtime_group;

static void enomem();

static void
copy_interval(dpu_runtime_interval *interval, struct timespec *const start,
              struct timespec *const end);

static int
alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows);
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

/** @brief alloc dpu set with given number of dpus */
void alloc_dpus(uint64_t nr_dpus);

void
populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data,
              dpu_runtime_totals *runtime);


dpu_error_t
post_process(struct dpu_set_t dpu_rank, uint32_t rank_id, void *arg);

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

/** @brief perform DPU lookup operation in embedding set and for input indices of
        multiple batch
    @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
    @param offsets array that stores indices offset (pytorch EmbedingBag convention) [EMB_INDEX][BATCH_INDEX][OFFSET]
    @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
    @param nr_batches gives the number of batch (same for each embedding) in indices
    @param final_results embedding lookup operation DPU results
    @return TBC
*/

int32_t *
lookup(uint32_t **indices, uint32_t **offsets, uint32_t *indices_len, uint32_t *nr_batches,
       float **final_results);