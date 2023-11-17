// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common.h"
#include "host/include/host.h"
#include "emb_types.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
// #include <sys/time.h>

#define RT_CONFIG 0

#ifndef DPU_BINARY
#    define DPU_BINARY "/home/jwong/PIM-Embedding-Lookup/upmem/build/release/dpu/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

#define NB_BLOCKS_POP_MRAM 1  // No. of blocks to transfer
#define NB_BLOCKS_LOOKUP 3
#define COL_PER_RANK 64

// REPLACE OLD - PLACEHOLDERS
#define NR_COLS -1					// self-note: clear (del later)
#define MAX_INDICES_PER_BATCH -1	// self-note: clear (del later)
#define MAX_NR_BATCHES -1			// self-note: clear (del later)
#define NR_TABLES -1				// self-note: clear (del later)

// // TODO: Change to dynamically allocated items in functions
uint32_t INDICES_LEN = MAX_INDICES_PER_BATCH*MAX_NR_BATCHES;

// int32_t* buffer_data[NR_COLS];	// Old - FIX TODO

bool first_run=true;
struct dpu_set_t *dpu_set;

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct dpu_runtime
 * @brief DPU execution times
 */
typedef struct dpu_runtime_totals {
	double execution_time_prepare;
	double execution_time_populate_copy_in;
	double execution_time_copy_in;
	double execution_time_copy_out;
	double execution_time_aggregate_result;
	double execution_time_launch;
} dpu_runtime_totals;

/**
 * @struct dpu_timespec
 * @brief ....
 */
typedef struct dpu_timespec {
	long tv_nsec;
	long tv_sec;
} dpu_timespec;

/**
 * @struct dpu_runtime_interval
 * @brief DPU execution interval
 */
typedef struct dpu_runtime_interval {
	dpu_timespec start;
	dpu_timespec stop;
} dpu_runtime_interval;

/**
 * @struct dpu_runtime_config
 * @brief ...
 */
typedef enum dpu_runtime_config {
	RT_ALL = 0,
	RT_LAUNCH = 1
} dpu_runtime_config;

/**
 * @struct dpu_runtime_group
 * @brief ...
 */
typedef struct dpu_runtime_group {
	unsigned int in_use;
	unsigned int length;
	dpu_runtime_interval *intervals;
} dpu_runtime_group;

static void enomem() {
	fprintf(stderr, "Out of memory\n");
	exit(ENOMEM);
}

static void copy_interval(dpu_runtime_interval *interval,
		struct timespec * const start,
		struct timespec * const end) {
	interval->start.tv_nsec = start->tv_nsec;
	interval->start.tv_sec = start->tv_sec;
	interval->stop.tv_nsec = end->tv_nsec;
	interval->stop.tv_sec = end->tv_sec;
}

// static int alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows) {
// 	// int tmp;

// 	for(int j=0; j<NR_COLS; j++){

// 		size_t sz = nr_rows*sizeof(int32_t);

// 		// printf("\nIter: %d,\nSize of sz: %lu\nStart alloc...\n", j, sz);
// 		// scanf("%d", &tmp);

// 		buffer_data[j] = (int32_t*)malloc(ALIGN(sz,8));
// 		if (buffer_data[j] == NULL) {
// 			return ENOMEM;
// 		}

// 		for(int k=0; k<nr_rows; k++){
// 			buffer_data[j][k] = table_data[k*NR_COLS+j];
// 		}

// 	}
// 	return 0;
// }

uint32_t get_table_id(uint32_t dpu_index, uint32_t* nr_cols, uint32_t nr_tables, uint32_t* index_note) {
	uint32_t table_id = 0;
	uint32_t working_dpu_ind = 0;
	for (uint32_t i = 0; i < nr_tables; i++) {
		// printf("In gettableid loop index %d, nrtable %d, i %d, working %d\n", dpu_index, i, nr_tables, working_dpu_ind);
		working_dpu_ind += nr_cols[i];
		if (working_dpu_ind > dpu_index) {
			break;
		}
		table_id += 1;
	}
	// printf("Post gettableid loop index %d\n", dpu_index);
	*index_note = working_dpu_ind;
	return table_id;
}

/*
Params:
0. table_id: embedding table number.
1. nr_rows: number of rows of the embedding table
2. NR_COLS: number of columns of the embedding table
3. table_data: a pointer of the size nr_rows*NR_COLS containing table's data
Result:
This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
corresponding table with the index of the first and last row held in each dpu.
*/

struct dpu_set_t* populate_mram(uint32_t table_id, uint64_t nr_rows, uint32_t col, int32_t *table_data, dpu_runtime_totals *runtime, uint8_t first){
	struct timespec start, end;

	// if(table_id>=AVAILABLE_RANKS){
	//     fprintf(stderr,"%d ranks available but tried to load table %dth",AVAILABLE_RANKS,table_id);
	//     exit(1);
	// }

	// TIME_NOW(&start);
	// if (alloc_buffers(table_id, table_data, nr_rows) != 0) {
	//     enomem();
	// }
	//TIME_NOW(&end);

	//if (runtime) runtime->execution_time_prepare += TIME_DIFFERENCE(start, end);

	//TIME_NOW(&start);

	struct dpu_set_t dpu;

	// if(first_run){
	if (first == 1) {
		dpu_set=(struct dpu_set_t*) malloc(sizeof(struct dpu_set_t));
		DPU_ASSERT(dpu_alloc(NR_COLS*NR_TABLES, NULL, dpu_set));
		DPU_ASSERT(dpu_load(*dpu_set, "/home/jwong/PIM-Embedding-Lookup/upmem/build/release/dpu/emb_dpu_lookup", NULL));
		first_run=false;
	}

	uint32_t len;
	uint8_t dpu_id,rank_id;

	DPU_FOREACH(*dpu_set, dpu, dpu_id){
		// if(dpu_id<(table_id+1)*NR_COLS && dpu_id>=table_id*NR_COLS){
		if(dpu_id == table_id * NR_COLS + col){
			// DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id-(table_id*NR_COLS)]));
			// DPU_ASSERT(dpu_prepare_xfer(dpu, &(table_data[((dpu_id-table_id*NR_COLS)*nr_rows)])));
			DPU_ASSERT(dpu_prepare_xfer(dpu, table_data));
		}
	}
	DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "emb_data", 0, ALIGN(nr_rows*sizeof(int32_t),8), DPU_XFER_DEFAULT));


	// for (int i = 0; i < NR_COLS; i++)
	//     free(buffer_data[i]);
	//TIME_NOW(&end);

	//if (runtime) runtime->execution_time_populate_copy_in += TIME_DIFFERENCE(start, end);

	return dpu_set;
	}

	typedef struct sg_xfer_context {
		//uint32_t table_id;
		uint32_t nr_tables;
		uint32_t* nr_rows;
		uint32_t* nr_cols; 
		//int32_t *table_data;
		int32_t **emb_tables;
		uint32_t total_cols;

	} sg_xfer_context;


	bool get_block_pop_mram(struct sg_block_info *out, uint32_t dpu_index,
			uint32_t block_index, void *args) {
		
		// printf("In get block pop mram with dpuid - %d, blockid - %d\n", dpu_index, block_index);
		if (block_index >= NB_BLOCKS_POP_MRAM) {
			return false;
		}
		// printf("pass\n");

		/* Unpack the arguments */
		sg_xfer_context *sc_args = (sg_xfer_context *) args;

		// Make sure we dont assign bad memory address
		if (dpu_index >= sc_args->total_cols) {
			return false;
		}

		uint32_t nr_tables = sc_args->nr_tables;
		int32_t **emb_tables  = sc_args->emb_tables;
		uint32_t* nr_cols = sc_args->nr_cols;
		uint32_t index_note;
		// printf("In get block pop mram with dpuid - %d, nr_tables = %d, cols = %d\n", dpu_index, nr_tables, nr_cols[0]);
		uint32_t table_id = get_table_id(dpu_index, nr_cols, nr_tables, &index_note);
		uint32_t nr_rows = sc_args->nr_rows[table_id]; 	

		// Calculate col index
		uint32_t col_index = nr_cols[table_id] - (index_note - dpu_index);

		// if (dpu_index == 0)
		// 	printf("POPSG GETBLK CHECK [0][266] = %d\n", emb_tables[0][266]);

		//One row per dpu.		
		//currently assumed nr_rows and NR_COLs is constant for all DPUs. 
		//Form a relation between dpu_index and table_id 	
		//Each dpu_index holds one coloumn with one block of data (nr_rows*int32_t) 
		//nr_rows might vary for each table and should be handled later. 

		out->addr = (uint8_t *) &(emb_tables[table_id][col_index * nr_rows]);	
		out->length = ALIGN(nr_rows*sizeof(int32_t), 8);

		return true;
	}

	struct dpu_set_t* populate_mram_sg(uint32_t nr_tables, uint32_t* nr_rows, uint32_t* nr_cols, int32_t **emb_tables, dpu_runtime_totals *runtime) {

		struct timespec start, end;
		struct dpu_set_t dpu;

		// DEBUG
		int32_t **tmp = (int32_t**) emb_tables;
		// printf("EXPORT VAL CHK: emb_tables[0][0][0] = %d, emb_tables[0][%d][%d] = %d\n", 
		// 		tmp[0][0], 
		// 		nr_cols[0]-1, nr_rows[0]-1, tmp[0][nr_cols[0]*nr_rows[0]-1]);
		// printf("EXPORT VAL CHK: emb_tables[%d][0][0] = %d, emb_tables[%d][%d][%d] = %d\n", 
		// 		nr_tables-1, tmp[nr_tables-1][0], 
		// 		nr_tables-1, nr_cols[nr_tables-1], nr_rows[nr_tables-1]-1, tmp[nr_tables-1][nr_cols[nr_tables-1]*nr_rows[nr_tables-1]-1]);

		// Calculate total number of cols for total number of DPUs alloc'd
		uint32_t total_cols = 0;
		uint32_t max_rows = 0;
		for (uint32_t i = 0; i < nr_tables; i++) {
			if (nr_rows[i] > max_rows) {
				max_rows = nr_rows[i];
			}
			total_cols += nr_cols[i];
		}
		// printf("nr_cols, [0] = %d, [last] = %d, ptr = %lu\n", nr_cols[0], nr_cols[nr_tables-1], (uint64_t) nr_cols);

		dpu_set = (struct dpu_set_t*) malloc(sizeof(struct dpu_set_t));

		//Actual number of DPUs = nr_col1 + nr_col2 + nr_col3 + ... 
		DPU_ASSERT(dpu_alloc(total_cols,"sgXferEnable=true, sgXferMaxBlocksPerDpu=10", dpu_set));		// ??
		DPU_ASSERT(dpu_load(*dpu_set, "/home/jwong/PIM-Embedding-Lookup/upmem/build/release/dpu/emb_dpu_lookup", NULL));

		uint32_t len;
		uint8_t dpu_id, rank_id;


		//Scatter Gather call 
		sg_xfer_context sc_args = {
			.nr_tables = nr_tables,
			.nr_rows = nr_rows,
			.nr_cols = nr_cols,
			.emb_tables = emb_tables,
			.total_cols = total_cols,
		};

		// printf("POPSG CHECK [0][266] = %d\n", emb_tables[0][266]);

		get_block_t get_block_info = {.f = &get_block_pop_mram, .args = &sc_args, .args_size = sizeof(sc_args)};

		//The size that is transferred here to size transferred to each DPU, should the maximum block len, max (nr_rows)*sizeof(int32_t) 

		DPU_ASSERT(dpu_sync(*dpu_set));
		DPU_ASSERT(dpu_push_sg_xfer(*dpu_set, DPU_XFER_TO_DPU, "emb_data", 0,
					ALIGN(max_rows*sizeof(int32_t), 8),
					&get_block_info, (dpu_sg_xfer_flags_t) (DPU_SG_XFER_DISABLE_LENGTH_CHECK | DPU_SG_XFER_DEFAULT)));
		DPU_ASSERT(dpu_sync(*dpu_set));
		return dpu_set;

	}

	dpu_error_t post_process(struct dpu_set_t dpu_rank, uint32_t rank_id, void *args){
		struct callback_input* input = (struct callback_input*) args;

		// One rank = 64 cols, calculating table_id
		uint32_t max_rank_id = input->total_cols / COL_PER_RANK;
		uint32_t remain_cols = input->total_cols % COL_PER_RANK;
		// if (remain_cols == 0) max_rank_id -= 1;

		// printf("rank_id %d, max %d, remain %d\n", rank_id, max_rank_id, remain_cols);

		// // PIM: Profiling
		// struct timeval start;
		// struct timeval end;
		float** final_results = input->final_results;
		uint32_t* off_len = input->off_len;
		int32_t*** tmp_results = input->tmp_results;
		//int32_t*** tmp_results=input->tmp_results;
		// printf("before log read\n");
		dpu_error_t status = DPU_OK;
		//printf("inside callback:%d\n",rank_id);

		// long post_proc_lat;
		// clock_gettime(CLOCK_REALTIME, &start);

		// printf("C post_process(): nr_batches for %d = %d\n", rank_id, nr_batches[rank_id]);
		// printf("C post_process(): NR_COLS for %d = %d\n", rank_id, NR_COLS);

		//printf("C post_process(): Check tmp_results values:\n [%d][63][0] = %f, after div = %f\n", rank_id, (float)tmp_results[rank_id][63][63], (float)input->tmp_results[rank_id][63][63]/pow(10,9));
		if (rank_id <= max_rank_id) {
			uint8_t cols_to_iter = COL_PER_RANK;
			if (rank_id == max_rank_id) cols_to_iter = remain_cols;

			for (int col_ind = 0; col_ind < cols_to_iter; col_ind++) {

				uint32_t table_col_offset, all_table_col_ind = col_ind + rank_id * COL_PER_RANK;
				uint8_t table_id = get_table_id(all_table_col_ind, input->nr_cols, input->nr_tables, &table_col_offset);
				table_col_offset = input->nr_cols[table_id] - (table_col_offset - all_table_col_ind);

				for (int k = 0; k < input->off_len[table_id]; k++) {
					final_results[table_id][k * input->nr_cols[table_id] + table_col_offset] = (float) ((float) (tmp_results[table_id][table_col_offset][k]) / pow(10, 9));
					// if (table_id == 0 && table_col_offset == 0)
						// printf("batch = %d, tmp results = %d, final = %f\n", k, tmp_results[table_id][table_col_offset][k], final_results[table_id][k * input->nr_cols[table_id] + table_col_offset]);
				}
			}
		}
		
		// if (rank_id < NR_TABLES) {
		// 	for (int j = 0; j < 64; j++) {
		// 		for (int k = 0; k < nr_batches; k++)
		// 			final_results[rank_id][k * NR_COLS + j] = (float) input->tmp_results[rank_id][j][k] / pow(10,9);
		// 	}
		// }
		//printf("C post_process(): Check final_results values:\n [%d][63][0] = %f\n", rank_id, final_results[rank_id][63*NR_COLS+63]);

		//printf("C post_process(): done for rank_id = %d\n", rank_id);

		// clock_gettime(CLOCK_REALTIME, &end);
		// post_proc_lat = end.tv_sec*1000000 + end.tv_usec - start.tv_sec*1000000 - start.tv_usec;
		// printf("C: Post processing latency: %ld", post_proc_lat);

		return status;
	}

	/*
	Params:
	1. ans: a pointer that be updated with the rows that we lookup
	2. input: a pointer containing the specific rows we want to lookup
	3. length: contains the number of rows that we want to lookup from the table
	4. nr_rows: number of rows of the embedding table
	5. NR_COLS: number of columns of the embedding table
	Result:*
	This function updates ans with the elements of the rows that we have lookedup
	*/
	// int32_t* lookup(uint32_t** indices, uint32_t** offsets, float** final_results, void *dpu_set_ptr_untyped, int64_t latency_print
    //             //,dpu_runtime_group *runtime_group
    //             ){
	// 	// // Check env
	// 	// printf("C test: Check envs: NR_COLS=%d, NR_TABLES=%d, MAX_NR_BATCHES=%d, NR_TASKLETS=%d", NR_COLS, NR_TABLES, MAX_NR_BATCHES, NR_TASKLETS);
	// 	int latency_record = latency_print;
	// 	long ind_copy_lat, query_copy_lat, dpu_launch_lat, results_copy_lat, callback_prep_lat, wait_sync_lat;


	// 	// PIM: Profiling
	// 	struct timespec start, end;

	// 	//printf("starting lookup\n");
	// 	struct dpu_set_t *dpu_set_ptr = (struct dpu_set_t *) dpu_set_ptr_untyped;
	// 	// struct timespec start, end;
	// 	int dpu_id,table_id;
	// 	struct dpu_set_t dpu_rank,dpu, set;
	// 	struct query_len lengths[NR_TABLES];

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&start);
	// 	}

	// 	//if (runtime_group && RT_CONFIG == RT_ALL) TIME_NOW(&start);
	// 	DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
	// 		DPU_ASSERT(dpu_prepare_xfer(dpu,indices[(int)(dpu_id/NR_COLS)]));
	// 	}

	// 	DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
	// 		INDICES_LEN*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
	// 	//printf("copied indices\n");

	// 	DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
	// 		DPU_ASSERT(dpu_prepare_xfer(dpu,offsets[(int)(dpu_id/NR_COLS)]));
	// 	}
	// 	DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_offsets",0,ALIGN(
	// 		MAX_NR_BATCHES*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
	// 	//printf("copied offsets\n");

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		ind_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

	// 		TIME_NOW(&start);
	// 	}

	// 	DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
	// 		table_id=(int)(dpu_id/NR_COLS);
	// 		lengths[table_id].indices_len=INDICES_LEN;
	// 		lengths[table_id].nr_batches=MAX_NR_BATCHES;
	// 		DPU_ASSERT(dpu_prepare_xfer(dpu,&lengths[table_id]));
	// 	}
	// 	DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr,DPU_XFER_TO_DPU,"input_lengths",0,
	// 		sizeof(struct query_len),DPU_XFER_DEFAULT));
	// 	//printf("query copied\n");

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		query_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

	// 		TIME_NOW(&start);
	// 	}

	// 	DPU_ASSERT(dpu_launch(*dpu_set_ptr, DPU_SYNCHRONOUS));
	// 	//printf("launch done\n");

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		dpu_launch_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
	// 	}

	// 	int32_t ***tmp_results=(int32_t***)malloc(NR_TABLES*sizeof(int32_t**));
	// 	//printf("wanna copy\n");

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&start);
	// 	}

	// 	DPU_FOREACH(*dpu_set_ptr,dpu,dpu_id){
	// 		if(dpu_id%NR_COLS==0){
	// 			table_id=dpu_id/NR_COLS;
	// 			tmp_results[table_id]=(int32_t**)malloc(NR_COLS*sizeof(int32_t*));
	// 		}
	// 		tmp_results[table_id][dpu_id%NR_COLS]=(int32_t*)malloc(MAX_NR_BATCHES*sizeof(int32_t));
	// 		DPU_ASSERT(dpu_prepare_xfer(dpu,&tmp_results[table_id][dpu_id%NR_COLS][0]));
	// 	}
	// 	//printf("copying back\n");
	// 	DPU_ASSERT(dpu_push_xfer(*dpu_set_ptr, DPU_XFER_FROM_DPU, "results", 0, ALIGN(sizeof(int32_t)*MAX_NR_BATCHES,8), DPU_XFER_DEFAULT));
	// 	//printf("Copies done\n");

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		results_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
			
	// 		TIME_NOW(&start);
	// 	}

	// 	struct callback_input *callback_data=(struct callback_input*)malloc(sizeof(struct callback_input));
	// 	callback_data->final_results=final_results;
	// 	callback_data->nr_batches=MAX_NR_BATCHES;
	// 	callback_data->tmp_results=tmp_results;
	// 	//printf("callback input allocated\n");
		
	// 	DPU_ASSERT(dpu_callback(*dpu_set_ptr,post_process,(void*)callback_data,DPU_CALLBACK_ASYNC));
	// 	//printf("callback done4\n");
	// 	// DPU_FOREACH(*dpu_set_ptr, dpu) {
	// 	//     DPU_ASSERT(dpu_log_read(dpu, stdout));
	// 	// }

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		callback_prep_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
			
	// 		TIME_NOW(&start);
	// 	}

	// 	dpu_sync(*dpu_set_ptr);

	// 	if (latency_record == 1) {
	// 		TIME_NOW(&end);
	// 		wait_sync_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
	// 	}

	// 	// long dpu_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
	// 	// printf("DPU Lat: %ldμs\n", dpu_lat);
	// 	uint32_t instructions;
	// 	uint32_t clks_p_sec;
	// 	DPU_FOREACH(set, dpu) {
	// 		DPU_ASSERT(
	// 			dpu_copy_from(dpu, "instructions", 0, &instructions, sizeof(uint32_t)));
	// 	}
	// 	DPU_FOREACH(set, dpu) {
	// 		DPU_ASSERT(
	// 			dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clks_p_sec, sizeof(uint32_t)));
	// 	}
	// 	// printf("DPU Freq: %uHz\n", clks_p_sec);
	// 	// printf("DPU instructions: %u\n", instructions);
	// 	// printf("DPU IPC: %f\n", (float) instructions / (float) (clks_p_sec * dpu_lat * 1.0e-6));

	// 	//printf("sync done\n");
	// 	/* if (runtime_group && RT_CONFIG == RT_LAUNCH) {
	// 		if(runtime_group[table_id].in_use >= runtime_group[table_id].length) {
	// 			TIME_NOW(&end);
	// 			f//printf(stderr,
	// 				"ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
	// 				dpu_id, runtime_group[table_id].in_use, table_id, runtime_group[table_id].length);
	// 			exit(1);
	// 		}
	// 		copy_interval(
	// 			&runtime_group->intervals[runtime_group[table_id].in_use], &start, &end);
	// 			runtime_group[table_id].in_use++;
	// 	} */
	// 	/* free(callback_data);
	// 	for(int i=0; i<NR_TABLES; i++){
	// 		for(int j=0; j<NR_COLS; j++){
	// 			free(tmp_results[i][j]);
	// 		}
	// 		free(tmp_results[i]);
	// 	}
	// 	free(tmp_results); */

	// 	if (latency_record == 1) {
	// 		printf("C: Indices and offsets copying latency: %ldμs\n", ind_copy_lat);
	// 		printf("C: Query copying latency: %ldμs\n", query_copy_lat);
	// 		printf("C: Dpu launch latency: %ldμs\n", dpu_launch_lat);
	// 		printf("C: Results copy latency: %ldμs\n", results_copy_lat);
	// 		printf("C: Callback prep latency: %ldμs\n", callback_prep_lat);
	// 		printf("C: DPU sync latency: %ldμs\n", wait_sync_lat);
	// 	}
	// 	return 0;
	// }

	typedef struct sg_xfer_lookup_context {
		uint32_t** indices;
		uint32_t** offsets;
		struct query_len* lengths;
		uint32_t* nr_cols;
		uint32_t nr_tables;

	} sg_xfer_lookup_context;

	bool get_block_lookup(struct sg_block_info *out, uint32_t dpu_index,
			uint32_t block_index, void *args) {
		// printf("In get block lookup with dpuid - %d, block index - %d\n", dpu_index, block_index);

		if (block_index >= NB_BLOCKS_LOOKUP) {
			return false;
		}

		/* Unpack the arguments */
		sg_xfer_lookup_context *sc_args = (sg_xfer_lookup_context *) args;
		uint32_t** indices  = sc_args->indices;
		uint32_t** offsets = sc_args->offsets;
		struct query_len* lengths = sc_args->lengths;
		uint32_t* nr_cols = sc_args->nr_cols;
		uint32_t working_col, nr_tables = sc_args->nr_tables;
		// printf("Unpacked index %d, nr_tables = %d\n", dpu_index, nr_tables);

		// if (dpu_index == 0) {
			// printf("LOOKCOP indices[0][0] = %u, indices[0][1023] = %u\n", indices[0][0], indices[0][1023]);
        	// printf("LOOKCOP offsets[0][0] = %u, offsets[0][31] = %u\n", offsets[0][0], offsets[0][31]);
		// }

		// Calculate current table
		uint32_t table_id = get_table_id(dpu_index, nr_cols, nr_tables, &working_col);
		// printf("Got table id index %d\n", dpu_index);
		
		// working_col = nr_cols[table_id] - (working_col - dpu_index);
		switch (block_index) {
			case 0: 	// struct query_len
				out->length = sizeof(struct query_len);
				out->addr = (uint8_t*) (&(lengths[table_id]));
				// dpu_index == 0 ? printf("dpu %u, blk %u len %u\n", dpu_index, block_index, out->length) : 0;
				return true;
			case 1:		// Indices
				out->length = lengths[table_id].indices_len * sizeof(uint32_t);
				out->addr = (uint8_t*) indices[table_id];
				// dpu_index == 0 ? printf("dpu %u, blk %u len %u\n", dpu_index, block_index, out->length) : 0;
				return true;
			case 2:		// Offsets
				out->length = lengths[table_id].nr_batches * sizeof(uint32_t);
				out->addr = (uint8_t*) offsets[table_id];
				// dpu_index == 0 ? printf("dpu %u, blk %u len %u\n", dpu_index, block_index, out->length) : 0;
				return true;
			default:
				return false;
		}
		// printf("Addr and len set index %d\n", dpu_index);
		return true;
	}

	typedef struct sg_xfer_results_context {
		struct query_len* lengths;
		uint32_t* nr_cols;
		uint32_t nr_tables;
		int32_t*** tmp_results;

	} sg_xfer_results_context;

	bool get_block_results(struct sg_block_info *out, uint32_t dpu_index,
			uint32_t block_index, void* args) {
		if (block_index >= 1) {
			return false;
		}

		uint32_t index_note;
		sg_xfer_results_context *sc_args = (sg_xfer_results_context*) args;
		uint32_t table_id = get_table_id(dpu_index, sc_args->nr_cols, sc_args->nr_tables, &index_note);
		uint32_t col_index = sc_args->nr_cols[table_id] - (index_note - dpu_index);

		out->length = sc_args->lengths[table_id].nr_batches * sizeof(int32_t);
		out->addr = (uint8_t*) sc_args->tmp_results[table_id][col_index];
		return true;
		
		// printf("Setting addr for dpu %d, len %d, addr [%d][%d]\n", dpu_index, out->length, table_id, col_index);
	}

	int32_t* lookup_sg(uint32_t** indices, uint32_t** offsets, 
						uint32_t* ind_len, uint32_t* off_len, uint32_t nr_tables, 
						float** final_results, void *dpu_set_ptr_untyped, int64_t latency_print, uint32_t* nr_cols
			//,dpu_runtime_group *runtime_group
		       ){
		// // Check env
		// printf("C test: Check envs: NR_COLS=%d, NR_TABLES=%d, MAX_NR_BATCHES=%d, NR_TASKLETS=%d", NR_COLS, NR_TABLES, MAX_NR_BATCHES, NR_TASKLETS);
		int latency_record = latency_print;
		uint32_t largest_len = 0;
		uint32_t largest_batch = 0;
		uint32_t total_cols = 0;

		long ind_copy_lat, query_copy_lat, dpu_launch_lat, results_copy_lat, callback_prep_lat, wait_sync_lat;

		// printf("C++ LOOK indices[0][0] = %u, indices[0][%d] = %u\n", indices[0][0], ind_len[0] - 1, indices[0][ind_len[0] - 1]);
		// printf("C++ LOOK offsets[0][0] = %u, offsets[0][%d] = %u\n", offsets[0][0], off_len[0] - 1, offsets[0][off_len[0] - 1]);

		// printf("C++ LOOK indices[%d][0] = %u, indices[%d][%d] = %u\n", nr_tables-1, indices[nr_tables-1][0], nr_tables-1, ind_len[nr_tables-1]-1, indices[nr_tables-1][ind_len[nr_tables-1]-1]);
		// printf("C++ LOOK offsets[%d][0] = %u, offsets[%d][%d] = %u\n", nr_tables-1, offsets[nr_tables-1][0], nr_tables-1, off_len[nr_tables-1]-1, offsets[nr_tables-1][off_len[nr_tables-1]-1]);

		// printf("C++ IND LEN[0] = %d, OFF LEN[0] = %d\n", ind_len[0], off_len[0]);
		// printf("C++ IND LEN[LAST] = %d, OFF LEN[LAST] = %d\n", ind_len[nr_tables-1], off_len[nr_tables-1]);
		// printf("C++ COLS(batch)[0] = %d, COLS(batch)[LAST] = %d\n", nr_cols[0], nr_cols[nr_tables-1]);

		// PIM: Profiling
		struct timespec start, end;

		// printf("starting lookup\n");
		struct dpu_set_t *dpu_set_ptr = (struct dpu_set_t *) dpu_set_ptr_untyped;
		// struct timespec start, end;
		int dpu_id, table_id;
		struct dpu_set_t dpu_rank, dpu, set;

		struct query_len* lengths = (struct query_len*) malloc(nr_tables * sizeof(struct query_len));

		// printf("DEBUG: start calculating lengths\n");

		// Buffer is stroed as:
		// ||input_lengths| -- Indices -- | -- Offsets -- | PAD | -- results -- ||
		for (uint32_t i = 0; i < nr_tables; i++) {
			lengths[i].indices_len = ind_len[i];
			lengths[i].nr_batches = off_len[i];
			lengths[i].offsets_start =  sizeof(struct query_len) / sizeof(uint32_t) + ind_len[i];
			total_cols += nr_cols[i];

			// Get max copy size
			uint32_t total_len = lengths[i].indices_len + lengths[i].nr_batches;
			if (total_len > largest_len) {
				largest_len = total_len;
			}
			if (off_len[i] > largest_batch) {
				largest_batch = off_len[i];
			}
		}
		largest_len *= sizeof(uint32_t);
		largest_len += sizeof(struct query_len);
		// printf("len2 = %d\n", largest_len);

		// printf("DEBUG: done lengths calc\n");
		uint32_t results_offset = KILOBYTE(12) - largest_batch;
		for (uint32_t i = 0; i < nr_tables; i++) {
			lengths[i].results_start = results_offset;
		}
		// KILOBYTE(12) is the #int32_t in WRAM buffer in a DPU
		results_offset *= sizeof(int32_t);
		if (results_offset < ALIGN(largest_len, 8)) {
			printf("\nERROR: #Indices/#Offsets too large\n");
			return 0;
		}
		if (latency_record == 1) {
			TIME_NOW(&start);
		}

		sg_xfer_lookup_context sc_args = {
			.indices = indices,
			.offsets = offsets,
			.lengths = lengths,
			.nr_cols = nr_cols,
			.nr_tables = nr_tables,
		};

		get_block_t get_block_info = {.f = &get_block_lookup, .args = &sc_args, .args_size = sizeof(sc_args)};

		// printf("DEBUG: Right before push sg xfer, len = %d, nr_tables = %d\n", ALIGN(largest_len, 8), nr_tables);

		DPU_ASSERT(dpu_push_sg_xfer(*dpu_set_ptr, DPU_XFER_TO_DPU, "input_buffer", 0,
					ALIGN(largest_len, 8),
					&get_block_info, (dpu_sg_xfer_flags_t) (DPU_SG_XFER_DISABLE_LENGTH_CHECK | DPU_SG_XFER_DEFAULT)));
		DPU_ASSERT(dpu_sync(*dpu_set_ptr));
		// printf("launching dpus\n");
		DPU_ASSERT(dpu_launch(*dpu_set_ptr, DPU_SYNCHRONOUS));
		// printf("launch done\n");

		// if (latency_record == 1) {
		// 	TIME_NOW(&end);
		// 	dpu_launch_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
		// }

		int32_t ***tmp_results = (int32_t***) malloc(nr_tables * sizeof(int32_t**));
		//printf("wanna copy\n");

		if (latency_record == 1) {
			TIME_NOW(&start);
		}

		// printf("DEBUG: tmp results init malloc done\n");

		uint32_t working = 0;
		uint32_t col_index = 0;
		table_id = -1;

		// Change to simpler loop
		DPU_FOREACH(*dpu_set_ptr, dpu, dpu_id){
			if (dpu_id >= total_cols) break;

			if (dpu_id >= working) {
				table_id += 1;
				working += nr_cols[table_id];
				tmp_results[table_id] = (int32_t**) malloc(nr_cols[table_id] * sizeof(int32_t*));
			}

			col_index = nr_cols[table_id] - (working - dpu_id);
			tmp_results[table_id][col_index] = (int32_t*) malloc(off_len[table_id] * sizeof(int32_t));
		}

		// printf("DEBUG: tmp results full malloc done\n");

		sg_xfer_results_context sc_results_args = {
			.lengths = lengths,
			.nr_cols = nr_cols,
			.nr_tables = nr_tables,
			.tmp_results = tmp_results,
		};
		get_block_t get_block_info_results = {.f = &get_block_results, .args = &sc_results_args, .args_size = sizeof(sc_results_args)};

		
		DPU_ASSERT(dpu_sync(*dpu_set_ptr));
		// printf("DEBUG: right before xfer to final results, largestbatch = %u\n", largest_batch);
		DPU_ASSERT(dpu_push_sg_xfer(*dpu_set_ptr, DPU_XFER_FROM_DPU, "input_buffer", results_offset,
					ALIGN(largest_batch * sizeof(int32_t), 8),
					&get_block_info_results, (dpu_sg_xfer_flags_t) (DPU_SG_XFER_DISABLE_LENGTH_CHECK | DPU_SG_XFER_DEFAULT)));
					
		DPU_ASSERT(dpu_sync(*dpu_set_ptr));

		if (latency_record == 1) {
			TIME_NOW(&end);
			results_copy_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;

			TIME_NOW(&start);
		}

		// printf("DEBUG: Done copying results\n");
		
		// for (int a = 0; a < off_len[0]; a++) {
		// 	printf("see res [0][0][%u] = %d\n", a, tmp_results[0][0][a]);
		// }
		// printf("\n");

		struct callback_input* callback_data = (struct callback_input*) malloc(sizeof(struct callback_input));
		callback_data->final_results = final_results;
		callback_data->off_len = off_len;	// Update to dynamic
		callback_data->tmp_results = tmp_results;
		callback_data->nr_tables = nr_tables;
		callback_data->nr_cols = nr_cols;
		callback_data->total_cols = total_cols;

		// printf("DEBUG: Starting callback\n");
		DPU_ASSERT(dpu_callback(*dpu_set_ptr, post_process, (void*) callback_data, DPU_CALLBACK_DEFAULT));
		// printf("callback done4\n");

		if (latency_record == 1) {
			TIME_NOW(&end);
			callback_prep_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
			TIME_NOW(&start);
		}

		dpu_sync(*dpu_set_ptr);

		if (latency_record == 1) {
			TIME_NOW(&end);
			wait_sync_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
		}

		// for (int a = 0; a < off_len[0]; a++) {
		// 	printf("see res [0][0][%u] = %f\n", a, final_results[0][a * num_cols_global[0]]);
		// }

		// scanf("%d", &largest_batch);

		// long dpu_lat = end.tv_sec*1000000 + end.tv_nsec/1000 - start.tv_sec*1000000 - start.tv_nsec/1000;
		// printf("DPU Lat: %ldμs\n", dpu_lat);
		// uint32_t instructions;
		// uint32_t clks_p_sec;
		// DPU_FOREACH(set, dpu) {
		// 	DPU_ASSERT(
		// 			dpu_copy_from(dpu, "instructions", 0, &instructions, sizeof(uint32_t)));
		// }
		// DPU_FOREACH(set, dpu) {
		// 	DPU_ASSERT(
		// 			dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clks_p_sec, sizeof(uint32_t)));
		// }

		// printf("printing log for dpus:\n");
		// DPU_FOREACH(*dpu_set_ptr, dpu, dpu_id) {
		// 	if (dpu_id == 0)
		// 		DPU_ASSERT(dpu_log_read(dpu, stdout));
  		// }
		// scanf("%d", &largest_batch);

		if (latency_record == 1) {
			printf("C: Indices and offsets copying latency: %ldμs\n", ind_copy_lat);
			printf("C: Query copying latency: %ldμs\n", query_copy_lat);
			printf("C: Dpu launch latency: %ldμs\n", dpu_launch_lat);
			printf("C: Results copy latency: %ldμs\n", results_copy_lat);
			printf("C: Callback prep latency: %ldμs\n", callback_prep_lat);
			printf("C: DPU sync latency: %ldμs\n", wait_sync_lat);
		}

		// Free malloc'd items
		free(lengths);
		free(callback_data);
		for (uint16_t i = 0; i < nr_tables; i++) {
			for (uint32_t j = 0; j < nr_cols[i]; j++) {
				free(tmp_results[i][j]);
			}
			free(tmp_results[i]);
		}
		free(tmp_results);

		return 0;
	}
