#include "emb_host.h"

#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct dpu_set_t* dpu_set;

void synthetic_populate(uint32_t nr_rows, uint32_t nr_cols, uint32_t nr_tables){
	// dpu_runtime_totals testing;
	for (uint32_t k=0; k<nr_tables; k++){
		int32_t table_data[nr_rows*nr_cols];
		for (int i=0; i<nr_rows*nr_cols; i++)
			table_data[i]=(int)rand();
		dpu_set=populate_mram(k, nr_rows,table_data, NULL);
	}
}

float** synthetic_inference(uint32_t nr_tables, uint32_t nr_batches, uint32_t indices_per_batch,
 uint32_t nr_rows, uint32_t nr_cols){
	 printf("DEBUG: In synthetic_inference()\n");

	uint32_t** synthetic_indices=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t** synthetic_offsets=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t* synthetic_indices_len=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));
	uint32_t* synthetic_nr_batches=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));

	
	printf("DEBUG: synthetic_inference() - Done array prep.\n");


	float** final_results=(float**)malloc(nr_tables*sizeof(float*));
	for (int k=0; k<nr_tables; k++){
		synthetic_indices[k]=(uint32_t*)malloc(nr_batches*indices_per_batch*sizeof(uint32_t));
		synthetic_offsets[k]=(uint32_t*)malloc(nr_batches*sizeof(uint32_t));
		final_results[k]=(float*)malloc(nr_batches*nr_cols*sizeof(uint32_t));
		synthetic_indices_len[k]=nr_batches*indices_per_batch;
		synthetic_nr_batches[k]=nr_batches;
		for (int i=0; i<nr_batches; i++){
			synthetic_offsets[k][i]=i*indices_per_batch;
			for (int j=0; j<indices_per_batch; j++){
				synthetic_indices[k][i*indices_per_batch+j]=(uint32_t)((double)rand()/RAND_MAX*nr_rows);

			}
		}
	}

	printf("DEBUG: synthetic_inference() - Done loading synthetic data.\n");

	lookup(synthetic_indices, synthetic_offsets, synthetic_indices_len,
                synthetic_nr_batches, final_results,(void*)dpu_set);
	for (int k=0; k<nr_tables; k++){
		free(synthetic_indices[k]);
		free(synthetic_offsets[k]);
	}
	free(synthetic_indices_len);
	free(synthetic_nr_batches);

	return final_results;
}

int main(){
	/* NR_COLS   			= 32
	 * NR_ROWS  		 	= 10
	 * NR_TABLES			= 10
	 * NR_BATCHES 			= 32
	 * indices_per_batch 	= 32
	*/

	printf("DEBUG: Starting synthetic_populate()...\n");
	synthetic_populate(10,32,10);

	printf("DEBUG: Done synthetic_populate(), starting synthetic_inference()...\n");
	float** results=synthetic_inference(10,32,32,10,32);

	
	printf("DEBUG: Done synthetic_inference().\n");

}


