#include "emb_host.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NR_COLS   			= 64
// NR_ROWS  		 	= 50000
// NR_TABLES			= 1
// NR_BATCHES 			= 64
// indices_per_batch 	= 32

/** @brief global referene to dpu_set */
struct dpu_set_t *dpu_set;

/** @brief host side embedding table buffer */
static int32_t **emb_tables;

/** @brief number of inference to perform for a parameter set n times inference with given
 * parameters */

static uint64_t NUM_RUN = 100;
static uint64_t NR_BATCHES_ = MAX_NR_BATCHES;
static uint64_t INDEX_PER_BATCH = 32;
static uint64_t NR_ROWS = 50000;
static uint64_t NR_COLS_ = NR_COLS;

static uint64_t NR_EMBEDDING = 9;

static float **final_results;

/** @brief compute time difference from to timespec */
struct timespec
time_diff(struct timespec start, struct timespec end) {
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

/** @brief computes synthetic embedding tables and store it to DPU MRAM
    @param nr_rows Embedding Number of rows (same for each embedding)
    @param nr_cols Embedding Number of columns (same for each embedding)
    @param nr_embedding number of embedding in emb_tables
*/
void
synthetic_populate(uint32_t nr_rows, uint32_t nr_cols, uint32_t nr_embedding) {
    emb_tables = (int32_t **) malloc(nr_embedding * sizeof(int32_t *));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        /* allocate embedding table on host side */
        int32_t *table_data = (int32_t *) malloc(nr_rows * nr_cols * sizeof(int32_t));

        /* synthetize embedding table parameters */
        for (int i = 0; i < nr_rows * nr_cols; i++) {
            double data_norm = (double) (rand()) / RAND_MAX;
            table_data[i] = (int32_t) (UINT32_MAX * data_norm);
        }

        /* store one embedding to DPU MRAM */
        dpu_set = populate_mram(k, nr_rows, table_data, NULL);
        /* store one embedding to HOST BUFFER */
        emb_tables[k] = table_data;
        // free(table_data);
    }
}

/** @brief check DPU embedding inference result for each embedding and each batch
    @param emb_tables host side embeding tables
    @param nr_embedding number of embedding in emb_tables
    @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
    @param offsets array that stores indices offset (pytorch EmbedingBag convention)
   [EMB_INDEX][BATCH_INDEX][OFFSET]
    @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
    @param nr_batches gives the number of batch (same for each embedding) in indices
    @param nr_cols Embedding Number of columns (same for each embedding)
    @param results DPU embedding inference result buffer [EMB_INDEX][BATCH_INDEX * NR_COLS]
    @return host model result and DPU results are the same or not
*/
bool
check_embedding_set_inference(int32_t **emb_tables, uint32_t nr_embedding, uint32_t **indices,
                              uint32_t **offsets, uint32_t *indices_len, uint32_t nr_batches,
                              uint32_t nr_cols, float **results) {
    bool valid = true;
    int32_t tmp_result[nr_cols];
    uint32_t index = 0;

    /* for each embedding */
    for (int embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        int ind_ptr = 0;
        /* for each input batch of index */
        for (int batch_index = 0; batch_index < nr_batches; batch_index++) {
            /* reset tmb buffer */
            for (int col_index = 0; col_index < nr_cols; col_index++)
                tmp_result[col_index] = 0;
            /* check limits */
            while ((ind_ptr < offsets[embedding_index][batch_index + 1] &&
                    batch_index < nr_batches - 1) ||
                   (batch_index == nr_batches - 1 && ind_ptr < indices_len[embedding_index])) {

                /* solve ind_ptr */
                index = indices[embedding_index][ind_ptr];
                for (int col_index = 0; col_index < nr_cols; col_index++)
                    /*Embedding reduction mode : ADD */
                    tmp_result[col_index] +=
                        emb_tables[embedding_index][index * nr_cols + col_index];
                /* next indice */
                ind_ptr++;
            }
            /* ckeck the batch result */
            for (int col_index = 0; col_index < nr_cols; col_index++) {

                float dpu_result = results[embedding_index][batch_index * nr_cols + col_index];
                float host_result = tmp_result[col_index];
                // printf("%f %f \n", dpu_result  , host_result);
                /* check magnitude with arbitrary threshold */
                if (fabs(dpu_result * pow(10, 9) - host_result) > 1000)
                    valid = false;
            }
        }
    }
    return valid;
}

/** @brief perform DPU embedding table inference given input indices with multiple embedding and
   multiple batch
    @param nr_embedding number of embedding in emb_tables
    @param nr_batches gives the number of batch (same for each embedding) in indices
    @param indices_pet_batch numbr of indices per batch
    @param nr_rows Embedding Number of rows (same for each embedding)
    @param nr_cols Embedding Number of columns (same for each embedding)
*/
void
synthetic_inference(float **final_results, uint32_t nr_embedding, uint32_t nr_batches,
                    uint32_t indices_per_batch, uint32_t nr_rows, uint32_t nr_cols) {

    uint32_t **indices = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    uint32_t **offsets = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    uint32_t *indices_len = (uint32_t *) malloc(nr_embedding * sizeof(uint32_t));
    uint32_t *nr_batches_per_embedding = (uint32_t *) malloc(nr_embedding * sizeof(uint32_t));

    for (int k = 0; k < nr_embedding; k++) {
        indices[k] = (uint32_t *) malloc(nr_batches * indices_per_batch * sizeof(uint32_t));
        offsets[k] = (uint32_t *) malloc(nr_batches * sizeof(uint32_t));
        final_results[k] = (float *) malloc(nr_batches * nr_cols * sizeof(uint32_t));
        indices_len[k] = nr_batches * indices_per_batch;
        nr_batches_per_embedding[k] = nr_batches;
        for (int i = 0; i < nr_batches; i++) {
            offsets[k][i] = i * indices_per_batch;
            for (int j = 0; j < indices_per_batch; j++) {
                double index_norm = ((double) rand() / RAND_MAX);
                uint32_t index = (uint32_t) (nr_rows * index_norm);
                indices[k][i * indices_per_batch + j] = index;
                assert(index < nr_rows);
            }
        }
    }

    struct timespec start, end, latency;
    int sum = 0;
    for (int i = 0; i < NUM_RUN; i++) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        lookup(indices, offsets, indices_len, nr_batches_per_embedding, final_results,
               (void *) dpu_set);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        sum += time_diff(start, end).tv_nsec;
    }
    bool valid = check_embedding_set_inference(emb_tables, nr_embedding, indices, offsets,
                                               indices_len, nr_batches, nr_cols, final_results);
    printf("inference : median latency [ms]: %lf, OK ? %d \n", 1e-6 * (double) sum / NUM_RUN,
           (int) valid);
    for (int k = 0; k < nr_embedding; k++) {
        free(indices[k]);
        free(offsets[k]);
    }
    free(indices_len);
    free(nr_batches_per_embedding);
}

/** @brief synthetize embedding table, input indices and perform DPU embedding table */
int
main() {

    /* alloc final results buffer */
    {
        final_results = (float **) malloc(NR_EMBEDDING * sizeof(float *));
        for (int k = 0; k < NR_EMBEDDING; k++) {
            final_results[k] = (float *) malloc(NR_BATCHES_ * NR_COLS_ * sizeof(uint32_t));
        }
    }
    synthetic_populate(NR_ROWS, NR_COLS_, NR_EMBEDDING);
    synthetic_inference(final_results, NR_EMBEDDING, NR_BATCHES_, INDEX_PER_BATCH, NR_ROWS,
                        NR_COLS);
    free(final_results);
}
