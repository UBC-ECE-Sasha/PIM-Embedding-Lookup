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
    printf("Hi I am lookup DPU.\n");
    unsigned int nr_rows, nr_cols, index_len;

    nr_rows = row_size_input;
    nr_cols = col_size_input;
    index_len=index_len_input;

    mram_offset=DPU_MRAM_HEAP_POINTER;
    
    __dma_aligned float read_buf[MAX_SIZE*4], write_buf[MAX_SIZE*4];
    __dma_aligned uint8_t read_len;

    if(nr_rows*nr_cols+index_len>MAX_SIZE /sizeof(float)){
        for (unsigned int data_read = 0; data_read < nr_rows*nr_cols+index_len; data_read+=MAX_SIZE/sizeof(float)){
            mram_offset=DPU_MRAM_HEAP_POINTER;
            mram_offset+=data_read*sizeof(float);
            mram_read(mram_offset, (float *)&read_buf[data_read], MAX_SIZE);
        }
        //printf("here\n");
    }
    else
    {   
        read_len=(nr_rows*nr_cols+index_len)*sizeof(float);
        if(read_len%8!=0)
            read_len+=8-(read_len%8);
        mram_read(mram_offset, (float *)read_buf, read_len);
    }

    for (int i=0;i<index_len;i++)
        printf("%d\n",(uint32_t)read_buf[nr_rows*nr_cols+i]);
    
    for(unsigned int i=0; i< index_len; i++){
        for(unsigned int j=0; j< nr_cols; j++){
            write_buf[i*nr_cols+j]=read_buf[((uint32_t)read_buf[nr_rows*nr_cols+i])*nr_cols+j];
            printf("data[%d][%d]:%f\n",(uint32_t)read_buf[nr_rows*nr_cols+i],j,write_buf[i*nr_cols+j]);
        }
    } 

    mram_offset=DPU_MRAM_HEAP_POINTER;
    mram_offset += nr_rows*nr_cols*sizeof(float);
    mram_write((const float *)write_buf,mram_offset,index_len*nr_cols*sizeof(float));

    return 0;
}
