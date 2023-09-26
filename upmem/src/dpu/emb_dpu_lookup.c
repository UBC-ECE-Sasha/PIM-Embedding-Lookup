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

__mram_noinit int32_t emb_data[MEGABYTE(14)]; // 56MB total

// __mram_noinit uint32_t input_indices[MAX_INDICES_PER_BATCH*MAX_NR_BATCHES];
// __mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
// __mram_noinit int32_t results[MAX_NR_BATCHES];
// Buffer that holds indices, offsets, and results
// Length of each is stored in input_lengths at the start
// ||input_lengths| -- Indices -- | -- Offsets -- | PAD | -- results -- ||
// Results length = nr_batches[k] (nr_batches = #offsets)
// (Indices always start at [4])
__mram_noinit int32_t input_buffer[KILOBYTE(12)];  // 48KB

// Other variables are in WRAM
uint32_t indices_ptr[NR_TASKLETS];
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

uint32_t indices_len, nr_batches, offsets_start, results_start;
// uint32_t copied_indices;
uint32_t *indices, *offsets;
int32_t *tmp_results;
struct query_len lengths;
__dma_aligned int32_t buffer[KILOBYTE(12)]; // WRAM max 64KB, 48KB used here
// __dma_aligned int32_t debug_use[32];
// __dma_aligned struct query_len lengths;
// __dma_aligned uint32_t indices[MAX_INDICES_PER_BATCH*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];
// __dma_aligned int32_t tmp_results[MAX_NR_BATCHES];

__host uint8_t first_run = 1;
int main() {
    // Profiling
    perfcounter_config(COUNT_CYCLES, true);

    __dma_aligned int32_t read_buff[2];  
    sem_take(&first_run_sem);
    if (first_run == 1) {
        mem_reset();
        // copied_indices = 0;

        // Read Lengths first, then read again for the rest
        mram_read(input_buffer, buffer, ALIGN(sizeof(struct query_len), 8));
        lengths = ((struct query_len*) buffer)[0];

        indices_len = lengths.indices_len;
        nr_batches = lengths.nr_batches;
        offsets_start = lengths.offsets_start;
        results_start = lengths.results_start;
        uint32_t lengths_offset = sizeof(struct query_len) / sizeof(uint32_t);

        // Read the rest
        // printf("DPU post len copy len %u\n", ALIGN((indices_len + nr_batches) * sizeof(uint32_t), 8));

        // mram_read seems to only handle copying 32 variables at a time
        uint32_t num_iters = (indices_len + nr_batches) / 32;
        for (uint32_t i = 0; i < num_iters; i++) {
            mram_read((__mram_ptr void const*) ((uint64_t) input_buffer + sizeof(struct query_len) + 32 * sizeof(uint32_t) * i), 
                        &buffer[lengths_offset + 32 * i], 
                        32 * sizeof(uint32_t));
        }
        mram_read((__mram_ptr void const*) (uint64_t) input_buffer + sizeof(struct query_len) + 32 * sizeof(uint32_t) * num_iters,
                    &buffer[lengths_offset + num_iters * 32],
                    ((indices_len + nr_batches) % 32) * sizeof(uint32_t));
        // mram_read((__mram_ptr void const *) ((uint64_t) input_buffer + sizeof(struct query_len)), &(buffer[lengths_offset]), ALIGN((indices_len + nr_batches) * sizeof(uint32_t), 8));

        indices = (uint32_t*) ((uint64_t) buffer + sizeof(struct query_len));
        offsets = (uint32_t*) &(buffer[offsets_start]);
        tmp_results = &(buffer[results_start]);

        // printf("DPU indices[0][0] = %u, indices[0][1023] = %u\n", indices[0], indices[1023]);
        // for (uint32_t i = 0; i < 128; i++) {
        //     // printf("DPU indices[0][%u] = %u\n", i, indices[i]);
        //     printf("DPU buffer[%u] = %u\n", i, buffer[4+i]);
        // }
        // printf("DPU offsets[0][0] = %u, offsets[0][31] = %u\n", offsets[0], offsets[31]);
        // printf("DPU buffer[4] = %u, buffer[1027] = %u\n", buffer[4], buffer[1027]);
        // printf("DPU buffer[1028] = %u, buffer[1059] = %u\n", buffer[1028], buffer[1059]);

        // mram_read((__mram_ptr void const*) ((uint64_t) input_buffer + 16 + 32*4), debug_use, 32*4);
        // for (uint32_t i = 0; i < 32; i++) {
        //     printf("TEST[%u] = %d\n", i, debug_use[i]);
        // }

        // printf("ind_len = %d, off_len = %d, off_start = %d, res_start = %d\n", indices_len, nr_batches, offsets_start, results_start);

        // while (copied_indices < indices_len){
        //     mram_read(&input_indices[copied_indices], &indices[copied_indices],
        //     ALIGN(MIN(2048, (indices_len - copied_indices) * sizeof(uint32_t)), 8));
        //     copied_indices += 2048 / sizeof(uint32_t);
        // }
        // mram_read(input_offsets,offsets, ALIGN(nr_batches * sizeof(uint32_t), 8));
        first_run = 0;
    }
    sem_give(&first_run_sem);

    if (me() != 0)
        indices_ptr[me()] = offsets[me()];
    else
        indices_ptr[me()] = 0;

    uint32_t last_written = 0;

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
    // printf("Check emb_data[0], [10], [2047]: %d, %d, %d\n", emb_data[0], emb_data[10], emb_data[2047]);

    for (uint64_t i = me(); i < nr_batches; i += NR_TASKLETS){
        tmp_results[i] = 0;
        while ((i == nr_batches - 1 && indices_ptr[me()] < indices_len) || 
                (i < nr_batches - 1 && indices_ptr[me()] < offsets[i + 1])) {
            uint32_t ind = indices[indices_ptr[me()]];
            
            mram_read(&emb_data[ind], read_buff, 8);
            // tmp_results[i] += read_buff[((ind % 2) != 0)];
            tmp_results[i] += read_buff[ind % 2];

            // if (i == 0)
                // printf("dpu0 cumu %d, embval %d, ind %d\n", tmp_results[i], read_buff[ind % 2], ind);

            indices_ptr[me()]++;
        }

        if ((i - 1) % 32 == 0 || i == nr_batches - 1) {
            sem_take(&result_sem);
            mram_write(&tmp_results[last_written], &(input_buffer[results_start + last_written]), ALIGN((i - last_written) * sizeof(int32_t), 8));
            // for (uint32_t a = last_written; a < i + 1; a++) {
            //     // printf("dpu results[%d] = %d, %d\n", a, tmp_results[a], buffer[a + results_start]);
            // }
            last_written = i + 1;
            sem_give(&result_sem);
        }
        
        if (i + NR_TASKLETS < nr_batches) {
            indices_ptr[me()] = offsets[i + NR_TASKLETS];
        }
    }
    sem_take(&first_run_sem);
     if (first_run == 0) {
         first_run = 1;
     }
     sem_give(&first_run_sem);

    // Profiling
    instructions = perfcounter_get();
    return 0;
}