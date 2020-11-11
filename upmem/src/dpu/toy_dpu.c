// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "common/include/emb_types.h"

#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit struct embedding_buffer emb_buffer;
__mram_noinit uint64_t input_nr_indices;
__mram_noinit uint64_t input_nr_offsets;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[4096];
__mram_noinit uint32_t input_offsets[4096];
__mram_noinit struct lookup_result results[4096];

int
main() {
    uint32_t first_row = emb_buffer.first_row;
    uint32_t last_row = emb_buffer.last_row;
    uint64_t nr_batches=input_nr_offsets;
    uint32_t indices_ptr=0;

    for (uint64_t i=0; i< nr_batches-1; i++){
        results[i].id=i;
        results[i].is_complete=true;

        for (int j=0; j<NR_COLS; j++)
            results[i].data[j]=0;

        printf("processing %lu th batch\n",i);
        while (indices_ptr<input_offsets[i+1])
        {
            if(input_indices[indices_ptr]<=last_row && input_indices[indices_ptr]>=first_row)
                for (int j=0; j<NR_COLS; j++)
                        results[i].data[j]+=emb_data[input_indices[indices_ptr]*NR_COLS+j];
            else
                results[i].is_complete=false;
            indices_ptr++;
            } 
    }

    for (int j=0; j<NR_COLS; j++)
        results[nr_batches-1].data[j]=0;
    printf("processing last batch\n");
        while(indices_ptr<input_nr_indices){
            if(input_indices[indices_ptr]<=last_row && input_indices[indices_ptr]>=first_row)
                for (int j=0; j<NR_COLS; j++)
                    results[nr_batches-1].data[j]+=emb_data[input_indices[indices_ptr]*NR_COLS+j];
            else
                results[nr_batches-1].is_complete=false;
            
            indices_ptr++;
        }

    return 0;
}
