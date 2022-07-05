// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "emb_types.h"

#include <attributes.h>
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>

__mram_noinit struct query_len input_lengths;
__host uint64_t emb_nr_rows;
__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[MAX_INDEX_PER_BATCH * MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram_noinit int32_t results[MAX_NR_BATCHES];

BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t indices_len, nr_batches;
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[MAX_INDEX_PER_BATCH * MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];
__dma_aligned int32_t tmp_results[MAX_NR_BATCHES];

int
main() {
    // DBG print emb_data content
    // if (me() == 0) {
    //     for (uint64_t i = 0; i < emb_nr_rows; i++) {
    //         printf("emb [%lu] : %d\n", i, emb_data[i]); // get_emb_data(emb_data ,i));
    //     }
    // }
    // return 0;
    if(me()==0){
        mram_read(&input_lengths, &lengths, ALIGN(sizeof(struct query_len),8));
        indices_len=lengths.indices_len;
        nr_batches=lengths.nr_batches;

        mram_read(input_indices,indices,ALIGN(indices_len*sizeof(uint32_t),8));
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));

        // for(uint32_t i=offsets[0]; i<offsets[0]+10; i++){
        for(uint32_t i=0; i<10; i++){
            printf("emb_data: %d\n",emb_data[i]);
        }
    }
    barrier_wait(&my_barrier);

    for (uint64_t i=me(); i< nr_batches; i+=NR_TASKLETS){
        tmp_results[i]=0;
        uint32_t upper_bound = i==nr_batches-1 ? indices_len : offsets[i+1];
        for(uint32_t indices_ptr=offsets[i]; indices_ptr<upper_bound; indices_ptr++)
        {
            uint32_t ind = indices[indices_ptr];
            tmp_results[i]+=emb_data[ind];
        }
    }

    barrier_wait(&my_barrier);
    if(me()==0){
        mram_write(tmp_results,results,ALIGN(nr_batches*sizeof(int32_t),8));
    }

    return 0;
}