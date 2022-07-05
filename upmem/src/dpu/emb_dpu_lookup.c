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
#include <barrier.h>

__mram_noinit struct query_len input_lengths;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[32*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram int32_t results[MAX_NR_BATCHES];

uint32_t indices_ptr[NR_TASKLETS];
SEMAPHORE_INIT(result_sem,1);
BARRIER_INIT(my_barrier, NR_TASKLETS);
int query_share;

uint32_t indices_len, nr_batches;
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[32*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];

int
main() {
    uint32_t copy_index;  
    if(me()==0){
        mram_read(&input_lengths, &lengths, ALIGN(sizeof(struct query_len),8));
        indices_len=lengths.indices_len;
        nr_batches=lengths.nr_batches;
        query_share=(int)nr_batches/NR_TASKLETS;
    }
    barrier_wait(&my_barrier);

    if(me()==1)
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));
    
    if(me()>1){
        copy_index=(me()-2)*(2048/sizeof(uint32_t));
        if(copy_index<indices_len){
            mram_read(&input_indices[copy_index],&indices[copy_index],
                ALIGN(MIN(2048, (indices_len-copy_index)*sizeof(uint32_t)),8)); 
        }
    }
    barrier_wait(&my_barrier);
    
    indices_ptr[me()]=offsets[me()*query_share];
    int32_t tmp_results[2*(int)(MAX_NR_BATCHES/NR_TASKLETS)];
    __dma_aligned int32_t read_buff[2];
    uint32_t batch_ptr, index_ptr;
    batch_ptr=me()*query_share;
    
    while ( (me()!=NR_TASKLETS-1 && batch_ptr<(me()+1)*query_share) || 
    (me()==NR_TASKLETS && batch_ptr<nr_batches) )
    {   
        index_ptr=offsets[batch_ptr];
        while((batch_ptr<nr_batches-1 && index_ptr<offsets[batch_ptr+1]) ||
            (batch_ptr==nr_batches-1 && index_ptr<indices_len)){
                uint32_t ind = indices[index_ptr];
                mram_read(&emb_data[ind],read_buff,8);
                tmp_results[batch_ptr-me()*query_share]+=read_buff[((ind % 2) != 0)];
                index_ptr++;
            }
        batch_ptr++;
    }
    sem_take(&result_sem);
    if(me()!=NR_TASKLETS-1)
        mram_write(&tmp_results[0],&results[0], ALIGN(query_share*sizeof(int32_t),8));
    else
        mram_write(&tmp_results[0],&results[0], ALIGN((query_share+
            (nr_batches%NR_TASKLETS))*sizeof(int32_t),8));
    sem_give(&result_sem);
    return 0;
}
