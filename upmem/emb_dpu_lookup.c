#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_SIZE 512 //Must be a power of 2

// To build the code: dpu-upmem-dpurte-clang -o emb_dpu_lookup emb_dpu_lookup.c

__mram_noinit uint8_t row_size_input;
__mram_noinit uint8_t col_size_input;
__mram_noinit uint8_t index_len_input;

__mram_ptr __dma_aligned uint8_t *mram_offset;

int main() {
    //printf("Hi I am lookup DPU.\n");
    unsigned int nr_rows, nr_cols, index_len;

    nr_rows = row_size_input;
    nr_cols = col_size_input;
    index_len=index_len_input;

    mram_offset=DPU_MRAM_HEAP_POINTER;
    
    __dma_aligned uint32_t read_buf[MAX_SIZE*8], write_buf[MAX_SIZE];
    __dma_aligned uint8_t write_len,read_len;

    if(nr_rows*nr_cols+index_len>MAX_SIZE /sizeof(uint32_t)){
        for (unsigned int data_read = 0; data_read < nr_rows*nr_cols+index_len; data_read+=MAX_SIZE/sizeof(uint32_t)){
            mram_offset=DPU_MRAM_HEAP_POINTER;
            mram_offset+=data_read*sizeof(uint32_t);
            mram_read(mram_offset, (uint32_t *)&read_buf[data_read], MAX_SIZE);
        }
        //printf("here\n");
    }
    else
    {   
        read_len=(nr_rows*nr_cols+index_len)*sizeof(uint32_t);
        if(read_len%8!=0)
            read_len+=8-(read_len%8);
        mram_read(mram_offset, read_buf, read_len);
    }

    for (int i=0;i<index_len;i++)
        printf("%d\n",read_buf[nr_rows*nr_cols+i]);
    
    for(unsigned int i=0; i< index_len; i++){
        for(unsigned int j=0; j< nr_cols; j++){
            write_buf[i*nr_cols+j]=read_buf[read_buf[nr_rows*nr_cols+i]*nr_cols+j];
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
