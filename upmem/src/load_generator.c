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
uint32_t* nr_rows_arr;
uint32_t* nr_cols_arr;

struct timespec time_diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec) < 0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000 + end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

int32_t** synthetic_populate(uint32_t* nr_rows, uint32_t* nr_cols, uint32_t nr_tables) {
	printf("TESTING SG IMPL\n");

	// dpu_runtime_totals testing;
	emb_tables = (int32_t**) malloc(nr_tables * sizeof(int32_t*) );
	int32_t first = 1;
	for (uint32_t k = 0; k < nr_tables; k++) {
		int32_t* table_data = (int32_t*) malloc(nr_rows[k] * nr_cols[k] * sizeof(int32_t));
		for (int i = 0; i < nr_rows[k] * nr_cols[k]; i++)
			table_data[i] = (int32_t) ((double) rand() / RAND_MAX * 1000000);

		// // Old
		// dpu_set = populate_mram(k, nr_rows_arr[0], 0, table_data, NULL, first);
		// first = 0;
		// for (int i = 1; i < nr_cols; i++)
		// 	dpu_set = populate_mram(k, nr_rows_arr[0], i, table_data, NULL, first);

		emb_tables[k] = table_data;
		//free(table_data);
	}

	// struct dpu_set_t* populate_mram_sg(uint32_t nr_tables, uint32_t* nr_rows, uint32_t* nr_cols, int32_t **emb_tables, dpu_runtime_totals *runtime);
	dpu_set = populate_mram_sg(nr_tables, nr_rows, nr_cols, emb_tables, NULL);
	return emb_tables;
}

bool validate_result(int32_t** emb_tables, uint32_t nr_tables, uint32_t** indices, uint32_t** offsets, 
uint32_t* indices_len, uint32_t* nr_batches, uint32_t* nr_cols, float** results, uint32_t max_cols, uint32_t* nr_rows) {
	bool valid = true;
	printf("In validate result\n");
	// printf("results test validate start %d\n", results[0][0]);
	int32_t* tmp_result = (int32_t*) malloc(max_cols * sizeof(int32_t));
	printf("post tmp malloc\n");
	uint32_t index = 0;
	for (int i = 0; i < nr_tables; i++) {

		// printf("validating table - %d\n", i);

		for (int j = 0; j < nr_batches[i]; j++) {
			int ind_ptr = offsets[i][j];
			if (j == 0) ind_ptr = 0;

			for (int t = 0; t < nr_cols[i]; t++) {
				tmp_result[t] = 0;
			}

			// printf("at batch %d, i %d, j %d\n", j, i, j);
			// printf("nr_batches[i] = %d\n", nr_batches[i]);
			// printf("offsets[i][j+1] = %d\n", offsets[i][j+1]);
			// printf("indices_len[i] = %d\n", indices_len[i]);
			while ((j == nr_batches[i] - 1 && ind_ptr < indices_len[i]) ||
					(j < nr_batches[i] - 1 && ind_ptr < offsets[i][j + 1])) {

				// printf("going through ind_ptr %d\n", ind_ptr);
				index = indices[i][ind_ptr];

				for (int t = 0; t < nr_cols[i]; t++) {
					tmp_result[t] += emb_tables[i][nr_rows[i] * t + index];

					// if (t == 0 && j == 0) {
					// 	printf("loadgen0 cumu %d, embval %d, ind %d\n", tmp_result[t], emb_tables[i][nr_rows[i]*t+index], index);
					// }
					// printf("val %d\n", emb_tables[i][nr_rows[i] * t + index]);
					// tmp_result[t] += emb_tables[i][index * nr_cols[i] + t];
				}
				// printf("track 0: index = %d, final = %d, val = %d, accum = %d\n", index, index, emb_tables[i][index], tmp_result[0]);
				// printf("track 1: index = %d, final = %d, val = %d, accum = %d\n", index, index, emb_tables[i][nr_rows[i] + index], tmp_result[1]);
				

				ind_ptr++; 
			}
			// printf("DPU track0 accum res = %f\n", (float) results[i][j*nr_cols[i]+0]);
			// printf("DPU track1 accum res = %f\n", (float) results[i][j*nr_cols[i]+1]);
			// printf("after while loop at batch %d\n", j);
			for (int t = 0; t < nr_cols[i]; t++) {
				if (abs((int32_t) (results[i][j * nr_cols[i] + t] * pow(10, 9)) - tmp_result[t]) > 1000) {
					valid = false;
				// 	if (i == 0 && t == 0)
				// 		printf("Validate BAD - table %d, batch %d, col %d, DPU float %f, DPU %d, VALID %d\n", i, j, t, results[i][j * nr_cols[i] + t], (int32_t) (results[i][j * nr_cols[i] + t] * pow(10, 9)), tmp_result[t]);
				}
					
			}
				
		}
	}
	free(tmp_result);
	return valid;
}

float** synthetic_inference(uint32_t nr_tables, uint32_t *off_len, uint32_t* nr_rows, 
uint32_t* nr_cols, uint32_t **indices, uint32_t **offsets, uint32_t *ind_len) {
	float** final_results = (float**) malloc(nr_tables * sizeof(float*));
	printf("DEBUG: malloc'd init. arrays\n");

	for (int k = 0; k < nr_tables; k++) {
		// ind_len[k] = off_len[k] * ind_len[k];
		// off_len[k] = off_len[k];

		indices[k] = (uint32_t*) malloc(off_len[k] * ind_len[k] * sizeof(uint32_t));
		offsets[k] = (uint32_t*) malloc(off_len[k] * sizeof(uint32_t));
		final_results[k] = (float *) malloc(off_len[k] * nr_cols[k] * sizeof(uint32_t));

		for (int i = 0; i < off_len[k]; i++) {
			offsets[k][i] = i * 10;
			for (int j = 0; j < ind_len[k]; j++) {
				indices[k][i * ind_len[k] + j] = (uint32_t)((double) rand() / RAND_MAX * nr_rows[k]);
			}
		}
		ind_len[k] = off_len[k] * ind_len[k];

		// if (k == 0) {
		// 	printf("LOADGEN indices[0][0] = %u, indices[0][1023] = %u\n", indices[0][0], indices[0][1023]);
		// 	printf("LOADGEN offsets[0][0] = %u, offsets[0][31] = %u\n", offsets[0][0], offsets[0][31]);
		// }
	}
	printf("DEBUG: malloc'd and filled indices, offsets, final_results\n");

	//printf("DEBUG: synthetic_inference() - Done loading synthetic data.\n");
	struct timespec start, end, latency;
	int sum = 0;
	// for (int i = 0; i < 100; i++) {
	// 	printf("#%d calls of lookup_sg\n", i);
	// 	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	lookup_sg(indices, offsets, ind_len, off_len, nr_tables, final_results, (void*) dpu_set, 0, nr_cols);
	// 	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	// 	sum += time_diff(start, end).tv_nsec;
	// }

	//printf("median latency:%d\n", sum/100);

	/* bool valid = validate_result(emb_tables, nr_tables, synthetic_indices, synthetic_offsets, 
		synthetic_indices_len, nr_batches, nr_cols, final_results);
	printf("Validation result:");
	printf(valid ? "true\n" : "false\n"); */


	//printf("DEBUG: synthetic_inference() - Done lookup.\n");
	// for (int a = 0; a < off_len[0]; a++) {
	// 	printf("res load gen [0][0][%u] = %f\n", a, final_results[0][a * num_cols_global[0]]);
	// }
	// scanf("%d", &sum);

	return final_results;
}

int main() {
	uint32_t rows[5] = {2048, 1024, 4096, 8192, 2048};
	uint32_t cols[5] = {64, 64, 64, 32, 32};
	// uint32_t rows[5] = {2048, 2048, 2048, 2048, 2048};
	// uint32_t cols[5] = {64, 64, 64, 64, 64};
	uint32_t ind_len[5] = {32, 32, 32, 32, 32};
	uint32_t off_len[5] = {32, 32, 32, 32, 32};
	uint32_t max_cols = 64;
	uint32_t **indices, **offsets;
	uint32_t nr_tables = 5;

	printf("DEBUG: Starting synthetic_populate()...\n");
	struct timespec start, end;
	TIME_NOW(&start);
	int32_t** emb_tables = synthetic_populate(rows, cols, nr_tables);
	TIME_NOW(&end);
	long diff = end.tv_sec * 1000000 + end.tv_nsec / 1000 - start.tv_sec * 1000000 - start.tv_nsec / 1000;
	printf("Time: %ldus, or %ldms\n", diff, diff / 1000);

	printf("DEBUG: Done synthetic_populate(), starting synthetic_inference()...\n");
	indices = (uint32_t**) malloc(nr_tables * sizeof(uint32_t*));
	offsets = (uint32_t**) malloc(nr_tables * sizeof(uint32_t*));
	float** results = synthetic_inference(nr_tables, off_len, rows, cols, indices, offsets, ind_len);
	printf("DEBUG: Done synthetic_inference().\n");
	bool valid = validate_result(emb_tables, nr_tables, indices, offsets, ind_len, off_len, cols, results, max_cols, rows);
	printf("\nValidation result: ");
	printf(valid ? "true\n" : "false\n");

	for (int k = 0; k < nr_tables; k++) {
		free(indices[k]);
		free(offsets[k]);
	}
	free(indices);
	free(offsets);

}


