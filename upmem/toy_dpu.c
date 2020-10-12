// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common.h"

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t first_row_input;
__mram_noinit uint64_t last_row_input;


__mram_noinit __host int32_t emb_data[MEGABYTE(14)];
__mram_noinit int ans_buffer[MEGABYTE(1)];

int main() {

    uint64_t nr_rows= row_size_input;
    uint64_t nr_cols=col_size_input;
    uint64_t first_row=first_row_input;
    uint64_t last_row=last_row_input;
    uint64_t b=nr_rows+nr_cols+first_row+last_row;

    __dma_aligned int32_t first,last;

    printf("In dpu first: %d, %d\n",emb_data[0],emb_data[1]);
    ans_buffer[0]=emb_data[0];
    printf("In dpu last: %d, %d\n",emb_data[(last_row-first_row+1)*nr_cols-1],emb_data[(last_row-first_row+1)*nr_cols]);
    ans_buffer[1]=emb_data[(last_row-first_row+1)*nr_cols];


    return 0;
}