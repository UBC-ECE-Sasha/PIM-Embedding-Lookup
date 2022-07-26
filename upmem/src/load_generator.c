#include "emb_host.h"

#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct dpu_set_t* dpu_set;
int32_t** emb_tables;

struct timespec time_diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

void synthetic_populate(uint32_t nr_rows, uint32_t nr_cols, uint32_t nr_tables){
	// dpu_runtime_totals testing;
	emb_tables=(int32_t**)malloc(nr_tables*sizeof(int32_t*));
	for (uint32_t k=0; k<nr_tables; k++){
		int32_t* table_data=(int32_t*)malloc(nr_rows*nr_cols*sizeof(int32_t));
		for (int i=0; i<nr_rows*nr_cols; i++)
			table_data[i]=(int)rand();
		dpu_set=populate_mram(k, nr_rows,table_data, NULL);
		emb_tables[k]=table_data;
		//free(table_data);
	}
}

bool validate_result(int32_t** emb_tables, uint32_t nr_tables,uint32_t** indices, uint32_t** offsets, 
uint32_t* indices_len, uint32_t nr_batches, uint32_t nr_cols, float** results){
	bool valid=true;
	int32_t tmp_result[nr_cols];
	uint32_t index=0;
	for(int i=0; i<nr_tables; i++){
		int ind_ptr=0;
		for(int j=0; j<nr_batches; j++){
			for(int t=0; t<nr_cols; t++)
				tmp_result[t]=0;
			while((ind_ptr<offsets[i][j+1] && j<nr_batches) ||
				(j==nr_batches && ind_ptr<indices_len[i])){
				index=indices[i][ind_ptr];
				for(int t=0; t<nr_cols; t++)
					tmp_result[t]+=emb_tables[i][index*nr_cols+t];
				ind_ptr++;
			}
			for(int t=0; t<nr_cols; t++){
				if(abs(results[i][j*nr_cols+t]*pow(10,9)-tmp_result[t])>1000)
					valid=false;
			}
				
		}
	}
	return valid;
}

float** synthetic_inference(uint32_t nr_tables, uint32_t nr_batches, uint32_t indices_per_batch,
 uint32_t nr_rows, uint32_t nr_cols){
	//printf("DEBUG: In synthetic_inference()\n");

	uint32_t** synthetic_indices=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t** synthetic_offsets=(uint32_t**)malloc(nr_tables*sizeof(uint32_t*));
	uint32_t* synthetic_indices_len=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));
	uint32_t* synthetic_nr_batches=(uint32_t*)malloc(nr_tables*sizeof(uint32_t));

	
	//printf("DEBUG: synthetic_inference() - Done array prep.\n");


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

	//printf("DEBUG: synthetic_inference() - Done loading synthetic data.\n");
	struct timespec start, end, latency;
	int sum=0;
	for(int i=0; i<100; i++){
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		lookup(synthetic_indices, synthetic_offsets, synthetic_indices_len,
                synthetic_nr_batches, final_results,(void*)dpu_set);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
		sum+=time_diff(start, end).tv_nsec;
	}

	//printf("median latency:%d\n", sum/100);

	/* bool valid=validate_result(emb_tables, nr_tables, synthetic_indices, synthetic_offsets, 
		synthetic_indices_len, nr_batches, nr_cols, final_results);
	printf("Validation result:");
	printf(valid ? "true\n" : "false\n"); */


	//printf("DEBUG: synthetic_inference() - Done lookup.\n");
	for (int k=0; k<nr_tables; k++){
		free(synthetic_indices[k]);
		free(synthetic_offsets[k]);
	}
	free(synthetic_indices_len);
	free(synthetic_nr_batches);

	return final_results;
}

int main(){
	uint32_t NR_ROWS= 50000;
	uint32_t NR_BATCHES= 128;
	uint32_t INDICES_PER_BATCH= 32;

	//printf("DEBUG: Starting synthetic_populate()...\n");
	synthetic_populate(NR_ROWS,NR_COLS,NR_TABLES);

	//printf("DEBUG: Done synthetic_populate(), starting synthetic_inference()...\n");
	float** results=synthetic_inference(NR_TABLES,NR_BATCHES,INDICES_PER_BATCH,
		NR_ROWS,NR_COLS);
	//printf("DEBUG: Done synthetic_inference().\n");

}


