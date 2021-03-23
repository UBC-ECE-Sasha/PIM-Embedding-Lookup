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
__mram_noinit struct lookup_query input_query;

__mram_noinit int32_t emb_data[MAX_CAPACITY];
__mram_noinit struct lookup_result results[MAX_NR_BATCHES];
__mram_noinit uint64_t first_run;
__mram_ptr uint8_t *mram_ptr;
__mram_noinit uint64_t nr_offsets;


uint32_t indices_ptr[NR_TASKLETS],first_row,last_row;
struct embedding_buffer table;
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

int rem_query_len;
uint8_t *wram_ptr;
__dma_aligned struct lookup_query query;

int
main() {
    sem_take(&first_run_sem);
    if(first_run==1){
        uint64_t tmp_nr_offsets;
        mem_reset();
        mram_read(&emb_buffer, &table, ALIGN(sizeof(struct embedding_buffer),8));
        first_row = table.first_row;
        last_row = table.last_row;

        wram_ptr=(uint8_t*)&query;
        mram_ptr=(__mram_ptr uint8_t*)&input_query;
        rem_query_len=sizeof(struct lookup_query);
        while(rem_query_len>0){
            mram_read(mram_ptr,wram_ptr,ALIGN(MIN(2048, rem_query_len),8));
            wram_ptr+=2048;
            mram_ptr+=2048;
            rem_query_len-=2048;
        }
        first_run=0;
        tmp_nr_offsets=((struct lookup_query)query).nr_offsets;
        mram_write(&tmp_nr_offsets,&nr_offsets,sizeof(uint64_t));
    }
    sem_give(&first_run_sem);
    
    __dma_aligned int32_t read_buff[ALIGN(NR_COLS+1,8)]; 
    __dma_aligned struct lookup_result tmp_result;  

    if(me()!=0)
        indices_ptr[me()]=((struct lookup_query)query).offsets[me()];
    else
        indices_ptr[me()]=0;

    for (uint32_t i=me(); i< ((struct lookup_query)query).nr_offsets; i+=NR_TASKLETS){
       
        tmp_result.id=i;
        tmp_result.is_complete=true;
        for (int j=0; j<NR_COLS; j++)
            tmp_result.data[j]=0;

       while ( (i==((struct lookup_query)query).nr_offsets-1 && indices_ptr[me()]<((struct lookup_query)query).nr_indices) || 
        (i<((struct lookup_query)query).nr_offsets-1 && indices_ptr[me()]<((struct lookup_query)query).offsets[i+1]) )
        {
            if(((struct lookup_query)query).indices[indices_ptr[me()]]<=last_row && 
            ((struct lookup_query)query).indices[indices_ptr[me()]]>=first_row){
                uint32_t ind = (((struct lookup_query)query).indices[indices_ptr[me()]]-first_row)*NR_COLS;
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

        if(i+NR_TASKLETS<((struct lookup_query)query).nr_offsets){
            indices_ptr[me()]=((struct lookup_query)query).offsets[i+NR_TASKLETS];
        }
    }
    return 0;
}