#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_SIZE 512 //Must be a power of 2

// To build the code: dpu-upmem-dpurte-clang -o emb_dpu_lookup emb_dpu_lookup.c

__mram_noinit uint32_t row_size_input;
__mram_noinit uint32_t col_size_input;
__mram_noinit uint32_t index_len_input;

__mram_ptr __dma_aligned uint8_t *mram_offset;

int main() {
    //printf("Hi I am lookup DPU.\n");
    unsigned int nr_rows, nr_cols, index_len;

    nr_rows = row_size_input;
    nr_cols = col_size_input;
    index_len=index_len_input;

    mram_offset=DPU_MRAM_HEAP_POINTER;
    
    __dma_aligned uint32_t read_buf[MAX_SIZE], write_buf[MAX_SIZE];
    __dma_aligned uint32_t write_len,read_len;

    read_len=index_len*sizeof(uint32_t);
    mram_offset += nr_rows*nr_cols*sizeof(uint32_t);

    if(read_len%8!=0)
        read_len+=8-(read_len%8);

    mram_read(mram_offset, read_buf, read_len);

    for (int i=0;i<index_len;i++)
        printf("%d\n",read_buf[i]);

    uint32_t __mram_ptr * emb=DPU_MRAM_HEAP_POINTER;
    
    for(unsigned int i=0; i< index_len; i++){
        for(unsigned int j=0; j< nr_cols; j++){
            write_buf[i*nr_cols+j]=emb[read_buf[i]*nr_cols+j];
            //printf("data[%d][%d]:%d\n",read_buf[nr_rows*nr_cols+i],j,write_buf[i*nr_cols+j]);
        }
    } 

    mram_offset=DPU_MRAM_HEAP_POINTER;
    mram_offset += nr_rows*nr_cols*sizeof(uint32_t);
    if((unsigned int)mram_offset%8!=0)
        mram_offset+=8-((unsigned int)mram_offset%8);
    
    write_len=index_len*nr_cols*sizeof(uint32_t);
    if(write_len%8!=0)
            write_len+=8-(write_len%8);
    
    mram_write((const uint32_t*)write_buf,mram_offset,write_len);

    return 0;
}
