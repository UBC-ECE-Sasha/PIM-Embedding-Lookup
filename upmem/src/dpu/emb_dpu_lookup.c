// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "emb_types.h"

#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <sem.h>

__mram_noinit struct buffer_meta emb_buffer;
__mram_noinit struct query_len input_lengths;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[32*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram_noinit int32_t results[MAX_NR_BATCHES];

uint32_t indices_ptr[NR_TASKLETS];
__dma_aligned struct buffer_meta table;
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

uint32_t indices_len, nr_batches, copied_indices;
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[32*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];
__dma_aligned int32_t tmp_results[MAX_NR_BATCHES];

__host uint8_t first_run = 1;
int
main() {
   /* __dma_aligned int32_t read_buff[8]; 
    mram_read(&emb_data[0],&read_buff[0],32);
    for (int i=0; i<4; i++)
        printf("%d\n",read_buff[i]);*/
    /*for (int i=0; i<8; i++)
        printf("%d, ",read_buff[i]);
    printf("\n------------\n"); */
    sem_take(&first_run_sem);
    if(first_run==1){
        mem_reset();
        copied_indices=0;

        mram_read(&emb_buffer, &table, ALIGN(sizeof(struct buffer_meta),8));

        mram_read(&input_lengths, &lengths, sizeof(struct query_len));
        indices_len=lengths.indices_len;
        nr_batches=lengths.nr_batches;

        while(copied_indices<indices_len){
            mram_read(&input_indices[copied_indices],&indices[copied_indices],
            ALIGN(MIN(2048, (indices_len-copied_indices)*sizeof(uint32_t)),8));
            copied_indices+=2048/sizeof(uint32_t);
        }
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));
        first_run=0;
    }
    sem_give(&first_run_sem);
    
    __dma_aligned int32_t read_buff[2];       

    if(me()!=0)
        indices_ptr[me()]=offsets[me()];
    else
        indices_ptr[me()]=0;

    uint32_t last_written=0;

    for (uint64_t i=me(); i< nr_batches; i+=NR_TASKLETS){

        tmp_results[i]=0;
        while ( (i==nr_batches-1 && indices_ptr[me()]<indices_len) || 
        (i<nr_batches-1 && indices_ptr[me()]<offsets[i+1]) )
        {
            uint32_t ind = indices[indices_ptr[me()]];
            mram_read(&emb_data[ind],read_buff,8);
            tmp_results[i]+=read_buff[((ind % 2) != 0)];
            indices_ptr[me()]++;
        }

        if((i-1)%512==0 || i==nr_batches-1){
            sem_take(&result_sem);
            mram_write(&tmp_results[last_written],&results[last_written], ALIGN(i*sizeof(int32_t),8));
            last_written=i+1;
            sem_give(&result_sem);
        }
        
        if(i+NR_TASKLETS<nr_batches){
            indices_ptr[me()]=offsets[i+NR_TASKLETS];
        }
    }
    sem_take(&first_run_sem);
    if(first_run==0){
        first_run=1;
    }
    sem_give(&first_run_sem);
    return 0;
}