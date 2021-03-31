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

__mram_noinit struct embedding_buffer emb_buffer;
__mram_noinit uint64_t input_nr_indices;
__mram_noinit uint64_t input_nr_offsets;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[32*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram_noinit struct lookup_result results[MAX_NR_BATCHES];
__mram_noinit uint64_t first_run;


uint32_t indices_ptr[NR_TASKLETS],first_row,last_row;
struct embedding_buffer table;
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

uint64_t nr_batches, indices_len, copied_indices;
__dma_aligned uint32_t indices[32*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];

int
main() {
    sem_take(&first_run_sem);
    if(first_run==1){
        mem_reset();
        copied_indices=0;

        mram_read(&emb_buffer, &table, ALIGN(sizeof(struct embedding_buffer),8));
        first_row = table.first_row;
        last_row = table.last_row;

        mram_read(&input_nr_indices, &indices_len, sizeof(uint64_t));
        mram_read(&input_nr_offsets, &nr_batches, sizeof(uint64_t));

        while(copied_indices<indices_len){
            mram_read(&input_indices[copied_indices],&indices[copied_indices],
            ALIGN(MIN(2048, (indices_len-copied_indices)*sizeof(uint32_t)),8));
            copied_indices+=2048/sizeof(uint32_t);
        }
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));
        first_run=0;
    }
    sem_give(&first_run_sem);
    
    __dma_aligned int32_t read_buff[ALIGN(NR_COLS+1,8)];       

    if(me()!=0)
        indices_ptr[me()]=offsets[me()];
    else
        indices_ptr[me()]=0;

    for (uint64_t i=me(); i< nr_batches; i+=NR_TASKLETS){
       
        __dma_aligned struct lookup_result tmp_result;
        tmp_result.id=i;
        tmp_result.is_complete=true;
        for (int j=0; j<NR_COLS; j++)
            tmp_result.data[j]=0;

        while ( (i==nr_batches-1 && indices_ptr[me()]<indices_len) || 
        (i<nr_batches-1 && indices_ptr[me()]<offsets[i+1]) )
        {
            if(indices[indices_ptr[me()]]<=last_row && indices[indices_ptr[me()]]>=first_row){
                uint32_t ind = (indices[indices_ptr[me()]]-first_row)*NR_COLS;
                mram_read(&emb_data[ind],read_buff,ALIGN(NR_COLS*sizeof(int32_t),8));
                for (int j=0; j<NR_COLS; j++) {
                    // Read from j + ((ind % 2) != 0) when read was from non-aligned address
                    tmp_result.data[j]+=read_buff[j+((ind % 2) != 0)];
                }
            }
            else{
                tmp_result.is_complete=false;
            }
            indices_ptr[me()]++;
        }

        sem_take(&result_sem);
        mram_write(&tmp_result,&results[i], sizeof(struct lookup_result));
        sem_give(&result_sem);

        if(i+NR_TASKLETS<nr_batches){
            indices_ptr[me()]=offsets[i+NR_TASKLETS];
        }
    }
    return 0;
}