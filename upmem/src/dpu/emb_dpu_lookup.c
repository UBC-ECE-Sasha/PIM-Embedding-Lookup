// To build the code: dpu-upmem-dpurte-clang -o emb_dpu_lookup emb_dpu_lookup.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_SIZE 128 //Must be a power of 2

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t index_len_input;

__mram_noinit int32_t data[8];

__mram_ptr __dma_aligned uint8_t *mram_offset;

int
main() {

    uint64_t nr_rows, nr_cols, index_len;

    nr_rows = row_size_input;
    nr_cols = col_size_input;
    index_len = index_len_input;

    mram_offset = DPU_MRAM_HEAP_POINTER;

    __dma_aligned uint64_t read_buf[MAX_SIZE];
    __dma_aligned double write_buf[MAX_SIZE];
    __dma_aligned uint64_t write_len, read_len;

    read_len = index_len * sizeof(uint64_t);

    mram_offset += nr_rows * nr_cols * sizeof(uint64_t);

    // updating the contents of read_buf with the index of the rows that we will lookup
    mram_read(mram_offset, read_buf, read_len);

    for (uint64_t i = 0; i < index_len; i++) {

        mram_offset = DPU_MRAM_HEAP_POINTER;

        mram_offset += read_buf[i] * nr_cols * sizeof(uint64_t);

        read_len = nr_cols * sizeof(uint64_t);

        // write the elemnts in row read_buf[i] of the table from write_buf[i*nr_cols] to from
        // write_buf[(i+1)*nr_cols-1]
        mram_read(mram_offset, write_buf + i * nr_cols, read_len);
    }

    mram_offset = DPU_MRAM_HEAP_POINTER;
    mram_offset += nr_rows * nr_cols * sizeof(uint64_t);

    write_len = index_len * nr_cols * sizeof(uint64_t);

    // write the contents of the write_buf in MRAM, replacing the contents of read_buf
    mram_write((const uint64_t *) write_buf, mram_offset, write_len);

    return 0;
}
