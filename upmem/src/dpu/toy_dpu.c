// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "emb_types.h"

#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit struct embedding_buffer emb_buffer;
__mram_noinit uint64_t input_nr_indices;
__mram_noinit uint64_t input_nr_offsets;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[1024];
__mram_noinit uint32_t input_offsets[1024];
__mram_noinit struct lookup_result results[1024];

int
main() {
    uint32_t first_row = emb_buffer.first_row;
    uint32_t last_row = emb_buffer.last_row;
    uint64_t nr_batches, indices_len;
    uint32_t indices_ptr=0;
    uint32_t indices[1024], offsets[1024];

    mram_read(&input_nr_indices, &indices_len, sizeof(uint64_t));
    mram_read(&input_nr_offsets, &nr_batches, sizeof(uint64_t));
    mram_read(input_indices,indices,ALIGN(indices_len*sizeof(uint32_t),8));
    mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));

    printf("indices_len=%lu and nr_batches=%lu\n",indices_len, nr_batches);

    for(uint64_t i=0; i<indices_len; i++)
        printf("indices[%lu]=%d\n",i,indices[i]);

    for(uint64_t i=0; i<nr_batches; i++)
        printf("offsets[%lu]=%d\n",i,offsets[i]);

    for (uint64_t i=0; i< nr_batches-1; i++){
        results[i].id=i;
        results[i].is_complete=true;
        for (int j=0; j<NR_COLS; j++)
            results[i].data[j]=0;

        printf("processing %lu th batch\n",i);
        while (indices_ptr<offsets[i+1])
        {
            if(indices[indices_ptr]<=last_row && indices[indices_ptr]>=first_row)
                for (int j=0; j<NR_COLS; j++)
                        results[i].data[j]+=emb_data[indices[indices_ptr]*NR_COLS+j];
            else
                results[i].is_complete=false;
            indices_ptr++;
        } 
        for (int j=0; j<NR_COLS; j++)
            results[i+1].data[j]=0;
        printf("processing last batch\n");
        while(indices_ptr<indices_len){
            if(indices[indices_ptr]<=last_row && indices[indices_ptr]>=first_row)
                for (int j=0; j<NR_COLS; j++)
                    results[i+1].data[j]+=emb_data[indices[indices_ptr]*NR_COLS+j];
            else
                results[i+1].is_complete=false;
            
            indices_ptr++;
        }
    }

    return 0;
}
