#include "common.h"
#include "host/include/host.h"
#include "emb_types.h"
#include "emb_host.c"

#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct dpu_set_t* dpu_set;

void synthetic_populate(uint32_t nr_rows, uint32_t nr_cols, uint32_t nr_tables){
	for (uint32_t k=0; k<nr_tables; k++){
		int32_t table_data[nr_rows*nr_cols];
		for (int i=0; i<nr_rows*nr_cols; i++)
			table_data[i]=(int)rand();
		dpu_set=populate_mram(k, nr_rows,table_data);
	}
}

float** synthetic_inference(uint32_t nr_tables, uint32_t nr_batches, uint32_t indices_per_batch,
 uint32_t nr_rows, uint32_t nr_cols){
	uint32_t** synthetic_indices=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t** synthetic_offsets=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t* synthetic_indices_len=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));
	uint32_t* synthetic_offsets_len=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));
	float** final_results=(float**)malloc(nr_tables*sizeof(float*));
	for (int k=0; k<nr_tables; k++){
		synthetic_indices[k]=(uint32_t*)malloc(nr_batches*indices_per_batch*sizeof(uint32_t));
		synthetic_offsets[k]=(uint32_t*)malloc(nr_batches*sizeof(uint32_t));
		final_results[k]=(float*)malloc(nr_batches*nr_cols*sizeof(uint32_t));
		synthetic_indices_len[k]=nr_batches*indices_per_batch;
		synthetic_offsets_len[k]=nr_batches;
		for (int i=0; i<nr_batches; i++){
			synthetic_offsets[k][i]=i*indices_per_batch;
			for (int j=0; j<indices_per_batch; j++){
				synthetic_indices[i][i*indices_per_batch+j]=(uint32_t)((double)rand()/RAND_MAX*nr_rows);

			}
		}
	}
	lookup(synthetic_indices, synthetic_offsets, synthetic_indices_len,
                synthetic_offsets_len, final_results,(void*)dpu_set);
	for (int k=0; k<nr_tables; k++){
		free(synthetic_indices[k]);
		free(synthetic_offsets[k]);
	}
	free(synthetic_indices_len);
	free(synthetic_offsets_len);

	return final_results;
}

int main(){

	synthetic_populate(10,64,10);
	float** results=synthetic_inference(10,32,32,10,64);
	
}


