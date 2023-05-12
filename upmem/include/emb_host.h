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
// #include <sys/time.h>

#define RT_CONFIG 0

#ifndef DPU_BINARY
#    define DPU_BINARY "/home/upmem0016/jwong5/PIM-Embedding-Lookup/upmem/build/release/dpu/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif


// PIM: Temp fix for bad env var passing... dont know why it stopped working
#define TMP_NR_TABLES 32

uint32_t INDICES_LEN = MAX_INDICES_PER_BATCH*MAX_NR_BATCHES;
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
    // int tmp;
    
    for(int j=0; j<NR_COLS; j++){

        size_t sz = nr_rows*sizeof(int32_t);

        // printf("\nIter: %d,\nSize of sz: %lu\nStart alloc...\n", j, sz);
        // scanf("%d", &tmp);

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

    // if(table_id>=AVAILABLE_RANKS){
    //     fprintf(stderr,"%d ranks available but tried to load table %dth",AVAILABLE_RANKS,table_id);
    //     exit(1);
    // }

    TIME_NOW(&start);
    if (alloc_buffers(table_id, table_data, nr_rows) != 0) {
        enomem();
    }
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_prepare += TIME_DIFFERENCE(start, end);

    //TIME_NOW(&start);

    struct dpu_set_t dpu;
    if(first_run){
        dpu_set=(struct dpu_set_t*)malloc(sizeof(struct dpu_set_t));
        DPU_ASSERT(dpu_alloc(NR_COLS*TMP_NR_TABLES, NULL, dpu_set));
        DPU_ASSERT(dpu_load(*dpu_set, DPU_BINARY, NULL));
        first_run=false;
    }

    uint32_t len;
    uint8_t dpu_id,rank_id;
    
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        if(dpu_id<(table_id+1)*NR_COLS && dpu_id>table_id*NR_COLS){
            DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id-(table_id*NR_COLS)]));
            // DPU_ASSERT(dpu_prepare_xfer(dpu, &(table_data[(dpu_id-(table_id*NR_COLS))])));
        }
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "emb_data", 0, ALIGN(nr_rows*sizeof(int32_t),8), DPU_XFER_DEFAULT));


    // for (int i = 0; i < NR_COLS; i++)
    //     free(buffer_data[i]);
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_populate_copy_in += TIME_DIFFERENCE(start, end);

    return dpu_set;
}


dpu_error_t post_process(struct dpu_set_t dpu_rank, uint32_t rank_id, void *arg){
    // // PIM: Profiling
    // struct timeval start;
    // struct timeval end;

    struct callback_input *input=(struct callback_input*)arg;
    float** final_results=input->final_results;
    uint32_t nr_batches=input->nr_batches;
    //int32_t*** tmp_results=input->tmp_results;
    // printf("before log read\n");
    dpu_error_t status=DPU_OK;
    //printf("inside callback:%d\n",rank_id);

    // long post_proc_lat;
    // clock_gettime(CLOCK_REALTIME, &start);

    // printf("C post_process(): nr_batches for %d = %d\n", rank_id, nr_batches[rank_id]);
    // printf("C post_process(): NR_COLS for %d = %d\n", rank_id, NR_COLS);

    //printf("C post_process(): Check tmp_results values:\n [%d][63][0] = %f, after div = %f\n", rank_id, (float)input->tmp_results[rank_id][63][63], (float)input->tmp_results[rank_id][63][63]/pow(10,9));

    if(rank_id<TMP_NR_TABLES){
        for (int j=0; j<NR_COLS; j++){
            for(int k=0; k<nr_batches; k++)
                final_results[rank_id][k*NR_COLS+j]=(float)input->tmp_results[rank_id][j][k]/pow(10,9);
        }
    }
    //printf("C post_process(): Check final_results values:\n [%d][63][0] = %f\n", rank_id, final_results[rank_id][63*NR_COLS+63]);

    //printf("C post_process(): done for rank_id = %d\n", rank_id);

    // clock_gettime(CLOCK_REALTIME, &end);
    // post_proc_lat = end.tv_sec*1000000 + end.tv_usec - start.tv_sec*1000000 - start.tv_usec;
    // printf("C: Post processing latency: %ld", post_proc_lat);

    return status;
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
int32_t* lookup(uint32_t** indices, uint32_t** offsets, float** final_results, void *dpu_set_ptr_untyped
                //,dpu_runtime_group *runtime_group
                ){
    // // Check env
    // printf("C test: Check envs: NR_COLS=%d, TMP_NR_TABLES=%d, MAX_NR_BATCHES=%d, NR_TASKLETS=%d", NR_COLS, TMP_NR_TABLES, MAX_NR_BATCHES, NR_TASKLETS);

    // // PIM: Profiling
    // struct timespec start, end;

    //printf("starting lookup\n");
    struct dpu_set_t *dpu_set_ptr = (struct dpu_set_t *) dpu_set_ptr_untyped;
    // struct timespec start, end;
    int dpu_id,table_id;
    struct dpu_set_t dpu_rank,dpu, set;
    struct query_len lengths[TMP_NR_TABLES];


    // long ind_copy_lat;
    // TIME_NOW(&start);

    //if (runtime_group && RT_CONFIG == RT_ALL) TIME_NOW(&start);
    DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu,indices[(int)(dpu_id/NR_COLS)]));
    }

    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
        INDICES_LEN*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
    //printf("copied indices\n");

    DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu,offsets[(int)(dpu_id/NR_COLS)]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_offsets",0,ALIGN(
        MAX_NR_BATCHES*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
    //printf("copied offsets\n");

    // TIME_NOW(&end);
    // ind_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

    // long query_copy_lat;
    // TIME_NOW(&start);

    DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
        table_id=(int)(dpu_id/NR_COLS);
        lengths[table_id].indices_len=INDICES_LEN;
        lengths[table_id].nr_batches=MAX_NR_BATCHES;
        DPU_ASSERT(dpu_prepare_xfer(dpu,&lengths[table_id]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_lengths",0,
        sizeof(struct query_len),DPU_XFER_DEFAULT));
    //printf("query copied\n");

    // TIME_NOW(&end);
    // query_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

    // long dpu_launch_lat;
    // TIME_NOW(&start);

    DPU_ASSERT(dpu_launch(*dpu_set_ptr, DPU_SYNCHRONOUS));
    //printf("launch done\n");

    // TIME_NOW(&end);
    // dpu_launch_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

    int32_t ***tmp_results=(int32_t***)malloc(TMP_NR_TABLES*sizeof(int32_t**));
    //printf("wanna copy\n");

    // long results_copy_lat;
    // TIME_NOW(&start);

    DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
        if(dpu_id%NR_COLS==0){
            table_id=dpu_id/NR_COLS;
            tmp_results[table_id]=(int32_t**)malloc(NR_COLS*sizeof(int32_t*));
        }
        tmp_results[table_id][dpu_id%NR_COLS]=(int32_t*)malloc(MAX_NR_BATCHES*sizeof(int32_t));
        DPU_ASSERT(dpu_prepare_xfer(dpu,&tmp_results[table_id][dpu_id%NR_COLS][0]));
    }
    //printf("copying back\n");
    DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr, DPU_XFER_FROM_DPU, "results", 0, ALIGN(sizeof(int32_t)*MAX_NR_BATCHES,8), DPU_XFER_DEFAULT));
    //printf("Copies done\n");

    // TIME_NOW(&end);
    // results_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
    
    // long callback_prep_lat;
    // TIME_NOW(&start);

    struct callback_input *callback_data=(struct callback_input*)malloc(sizeof(struct callback_input));
    callback_data->final_results=final_results;
    callback_data->nr_batches=MAX_NR_BATCHES;
    callback_data->tmp_results=tmp_results;
    //printf("callback input allocated\n");
    
    DPU_ASSERT(dpu_callback(*dpu_set_ptr,post_process,(void*)callback_data,DPU_CALLBACK_ASYNC));
    //printf("callback done4\n");
    // DPU_FOREACH(*dpu_set_ptr, dpu) {
    //     DPU_ASSERT(dpu_log_read(dpu, stdout));
    // }

    // TIME_NOW(&end);
    // callback_prep_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
    
    // long wait_sync_lat;
    // TIME_NOW(&start);

    dpu_sync(*dpu_set_ptr);

    // TIME_NOW(&end);
    // wait_sync_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
    
    // long dpu_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
    // printf("DPU Lat: %ldμs\n", dpu_lat);
    uint32_t instructions;
    uint32_t clks_p_sec;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(
            dpu_copy_from(dpu, "instructions", 0, &instructions, sizeof(uint32_t)));
    }
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(
            dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clks_p_sec, sizeof(uint32_t)));
    }
    // printf("DPU Freq: %uHz\n", clks_p_sec);
    // printf("DPU instructions: %u\n", instructions);
    // printf("DPU IPC: %f\n", (float) instructions / (float) (clks_p_sec * dpu_lat * 1.0e-6));

    //printf("sync done\n");
    /* if (runtime_group && RT_CONFIG == RT_LAUNCH) {
        if(runtime_group[table_id].in_use >= runtime_group[table_id].length) {
            TIME_NOW(&end);
            f//printf(stderr,
                "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                dpu_id, runtime_group[table_id].in_use, table_id, runtime_group[table_id].length);
            exit(1);
        }
        copy_interval(
            &runtime_group->intervals[runtime_group[table_id].in_use], &start, &end);
            runtime_group[table_id].in_use++;
    } */
    /* free(callback_data);
    for(int i=0; i<TMP_NR_TABLES; i++){
        for(int j=0; j<NR_COLS; j++){
            free(tmp_results[i][j]);
        }
        free(tmp_results[i]);
    }
    free(tmp_results); */

    
    // printf("C: Indices and offsets copying latency: %ldμs\n", ind_copy_lat);
    // printf("C: Query copying latency: %ldμs\n", query_copy_lat);
    // printf("C: Dpu launch latency: %ldμs\n", dpu_launch_lat);
    // printf("C: Results copy latency: %ldμs\n", results_copy_lat);
    // printf("C: Callback prep latency: %ldμs\n", callback_prep_lat);
    // printf("C: DPU sync latency: %ldμs\n", wait_sync_lat);
    return 0;
}