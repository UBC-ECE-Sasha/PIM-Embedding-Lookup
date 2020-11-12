// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common/include/common.h"
#include "emb_types.h"

#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit struct embedding_buffer emb_buffer;
__mram_noinit uint64_t input_nr_indices;
__mram_noinit uint64_t input_nr_offsets;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[4*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__host struct lookup_result results[MAX_NR_BATCHES];

int
main() {
    struct embedding_buffer table;
    mram_read(&emb_buffer, &table, ALIGN(sizeof(struct embedding_buffer),8));
    uint32_t first_row = table.first_row;
    uint32_t last_row = table.last_row;

    printf("first_row in dpu=%d and last_row in dpu=%d\n",first_row, last_row);

    uint64_t nr_batches, indices_len;
    uint32_t indices_ptr=0;
    uint32_t indices[1024], offsets[1024];
    int32_t tmp_buff[ALIGN(NR_COLS+1,8)];

    mram_read(&input_nr_indices, &indices_len, sizeof(uint64_t));
    mram_read(&input_nr_offsets, &nr_batches, sizeof(uint64_t));
    mram_read(input_indices,indices,ALIGN(indices_len*sizeof(uint32_t),8));
    mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));

    printf("input_nr_indices=%ld\n",indices_len);
    printf("input_nr_offsets=%ld\n",nr_batches);

    for(uint64_t i=0; i<indices_len; i++)
        printf("indices[%lu]=%d\n",i,indices[i]);

    for(uint64_t i=0; i<nr_batches; i++)
        printf("offsets[%lu]=%d\n",i,offsets[i]);

    for (uint64_t i=0; i< nr_batches-1; i++){
        results[i].id=i;
        results[i].is_complete=true;
        for (int j=0; j<NR_COLS; j++)
            results[i].data[j]=0;

        printf("processing %lu th batch\n",i);
        while (indices_ptr<offsets[i+1]) {
            if(indices[indices_ptr]<=last_row || indices[indices_ptr]>=first_row){
                uint32_t ind = indices[indices_ptr]*NR_COLS;
                mram_read(&emb_data[ind],tmp_buff,ALIGN(NR_COLS*sizeof(int32_t),8));
                for (int j=0; j<NR_COLS; j++) {
                    /* Read from j + ((ind % 2) != 0) when read was from non-aligned address */
                    results[i].data[j]+=tmp_buff[j+((ind % 2) != 0)];
                }
            }
            else
                results[i].is_complete=false;
            indices_ptr++;
        }
        for (int j=0; j<NR_COLS; j++)
            results[i+1].data[j]=0;
        printf("processing last batch\n");
        while(indices_ptr<indices_len){
            if(indices[indices_ptr]<=last_row || indices[indices_ptr]>=first_row){
                uint32_t ind = indices[indices_ptr]*NR_COLS;
                mram_read(&emb_data[ind],tmp_buff,ALIGN(NR_COLS*sizeof(int32_t),8));
                for (int j=0; j<NR_COLS; j++) {
                    /* Read from j + ((ind % 2) != 0) when read was from non-aligned address */
                    results[i+1].data[j]+=tmp_buff[j+((ind % 2) != 0)];
                }
            }
            else
                results[i+1].is_complete=false;

            indices_ptr++;
        }
    }

    return 0;
}
