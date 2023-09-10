#include <perfcounter.h>
#include <stdio.h>
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <sem.h>
#include "common/include/common.h"
#include "emb_types.h"

// #define TMP_MAX_NR_BATCHES 64
// #define TMP_MAX_INDICES_PER_BATCH 120

// Profiling
__host uint32_t instructions;

__mram_noinit struct query_len input_lengths;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[MAX_INDICES_PER_BATCH*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram_noinit int32_t results[MAX_NR_BATCHES];

uint32_t indices_ptr[NR_TASKLETS];
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

uint32_t indices_len, nr_batches, copied_indices;
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[MAX_INDICES_PER_BATCH*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];
__dma_aligned int32_t tmp_results[MAX_NR_BATCHES];

__host uint8_t first_run = 1;
int
main() {
    // Profiling
    perfcounter_config(COUNT_CYCLES, true);

    __dma_aligned int32_t read_buff[2];  
    sem_take(&first_run_sem);
    if(first_run==1){
        mem_reset();
        copied_indices=0;

        mram_read(&input_lengths, &lengths, ALIGN(sizeof(struct query_len),8));
        indices_len=lengths.indices_len;
        nr_batches=lengths.nr_batches;

        while(copied_indices<indices_len){
            mram_read(&input_indices[copied_indices],&indices[copied_indices],
            ALIGN(MIN(2048, (indices_len-copied_indices)*sizeof(uint32_t)),8));
            copied_indices+=2048/sizeof(uint32_t);
        }
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));
        first_run=0;
    }
    sem_give(&first_run_sem);

    if(me()!=0)
        indices_ptr[me()]=offsets[me()];
    else
         indices_ptr[me()]=0;

    uint32_t last_written=0;

    // DEBUG
    // printf("DPU Binary: Check indices:\n[ ");
    // for (int debug = 0; debug < 32*MAX_NR_BATCHES; debug++) {
    //     printf("%d, ", input_indices[debug]);
    // }
    // printf("]\n");
    // printf("DPU Binary: Check offsets:\n[ ");
    // for (int debug = 0; debug < MAX_NR_BATCHES; debug++) {
    //     if (input_offsets[debug] >= 32 * 64) {
    //         printf("OFFSET RANGE EXCEEDED");
    //     }
    //     // printf("%d, ", input_offsets[debug]);
    // }
    // printf("]\n");
    // printf("DPU Binary: Check indices_ptr:\n[ ");
    // for (int debug = 0; debug < NR_TASKLETS; debug++) {
    //     printf("%d, ", indices_ptr[debug]);
    // }
    // printf("]\n");
    // printf("DPU Binary: Check index reference by indices_ptr:\n[ ");
    // printf("indices_len = %d\n", indices_len);
    // for (uint32_t debug = 0; debug < indices_len; debug++) {
    //     if (indices[debug] >= 65000) {
    //         printf("INDEX RANGE EXCEEDED at %d: %u\n", debug, indices[debug]);
    //     }
    //     // printf("%d, ", indices[indices_ptr[debug]]);
    // }
    // printf("]\n");
    // printf("DPU Binary: Check embdata reference by index:\n[ ");
    // for (uint32_t debug = 0; debug < indices_len; debug++) {
    //     printf("%d, ", emb_data[indices[indices_ptr[debug]]]);
    // }
    // printf("%d, ", emb_data[0]);
    // printf("%d, ", emb_data[32000]);
    // printf("%d, ", emb_data[64999]);
    // printf("]\n");



    for (uint64_t i=me(); i< nr_batches; i+=NR_TASKLETS){

        tmp_results[i]=0;
        while ( (i==nr_batches-1 && indices_ptr[me()]<indices_len) || 
        (i<nr_batches-1 && indices_ptr[me()]<offsets[i+1]) )
        {
            uint32_t ind = indices[indices_ptr[me()]];
            mram_read(&emb_data[ind],read_buff,8);
            tmp_results[i]+=read_buff[((ind % 2) != 0)];
            indices_ptr[me()]++;
        }

        if((i-1)%512==0 || i==nr_batches-1){
            sem_take(&result_sem);
            mram_write(&tmp_results[last_written],&results[last_written], ALIGN(i*sizeof(int32_t),8));
            last_written=i+1;
            sem_give(&result_sem);
        }
        
        if(i+NR_TASKLETS<nr_batches){
            indices_ptr[me()]=offsets[i+NR_TASKLETS];
        }
    }
    sem_take(&first_run_sem);
     if(first_run==0){
         first_run=1;
     }
     sem_give(&first_run_sem);

    // Profiling
    instructions = perfcounter_get();
    return 0;
}