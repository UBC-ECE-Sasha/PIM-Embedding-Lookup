// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "common/include/emb_types.h"

#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit struct embedding_buffer emb_buffer;
__mram_noinit uint64_t input_indices_len;
__mram_noinit uint64_t input_offsets_len;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit int32_t input_indices[4096];
__mram_noinit int32_t input_offsets[4096];
__mram_noinit int ans_buffer[MEGABYTE(1)];

int
main() {
    uint64_t first_row = emb_buffer.first_row;
    uint64_t last_row = emb_buffer.last_row;

    printf("In dpu first elements: %d, %d\n", emb_data[0], emb_data[1]);
    ans_buffer[0] = emb_data[0];
    printf("In dpu last elements: %d, %d\n", emb_data[(last_row - first_row + 1) * NR_COLS - 2],
           emb_data[(last_row - first_row + 1) * NR_COLS - 1]);
    ans_buffer[1] = emb_data[(last_row - first_row + 1) * NR_COLS - 1];

    return 0;
}
