// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t first_index_input;
__mram_noinit uint64_t last_index_input;

__mram_ptr __dma_aligned uint8_t *mram_offset;

int main() {

    uint64_t nr_rows= row_size_input;
    uint64_t nr_cols=col_size_input;
    uint64_t first_index=first_index_input;
    uint64_t last_index=last_index_input;
    uint64_t b=nr_rows+nr_cols+first_index+last_index;

    __dma_aligned int32_t write_buf[2],read_buf[2];
    __dma_aligned int32_t first,last;

    mram_read(DPU_MRAM_HEAP_POINTER, read_buf, 2*sizeof(int32_t));
    printf("In dpu first: %d, %d\n",read_buf[0],read_buf[1]);
    write_buf[0]=read_buf[0];
    mram_offset+=(last_index-first_index-1)*sizeof(int32_t);
    mram_read(mram_offset, read_buf, 2*sizeof(int32_t));
    printf("In dpu last: %d, %d\n",read_buf[0],read_buf[1]);
    write_buf[1]=read_buf[1];

    mram_write((const int32_t*)write_buf, mram_offset+sizeof(int32_t), 2*sizeof(int32_t));


    return 0;
}