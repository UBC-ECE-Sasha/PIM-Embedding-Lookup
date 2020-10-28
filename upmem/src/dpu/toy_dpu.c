// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common/include/common.h"

__mram_noinit uint64_t nr_cols_input;
__mram_noinit uint64_t first_row_input;
__mram_noinit uint64_t last_row_input;
__mram_noinit uint64_t input_indices_len;
__mram_noinit uint64_t input_offsets_len;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit int32_t input_indices[4096];
__mram_noinit int32_t input_offsets[4096];
__mram_noinit int ans_buffer[MEGABYTE(1)];

int main() {

    uint64_t nr_cols= nr_cols_input;
    uint64_t first_row=first_row_input;
    uint64_t last_row=last_row_input;

    printf("In dpu first elements: %d, %d\n",emb_data[0],emb_data[1]);
    ans_buffer[0]=emb_data[0];
    printf("In dpu last elements: %d, %d\n",emb_data[(last_row-first_row+1)*nr_cols-2],emb_data[(last_row-first_row+1)*nr_cols-1]);
    ans_buffer[1]=emb_data[(last_row-first_row+1)*nr_cols-1];

    return 0;
}
