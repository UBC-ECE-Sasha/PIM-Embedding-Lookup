#include "embedding.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t NR_DPUS;

/* @brief final_results embedding lookup operationDPU results */
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
 *  @param nr_rows Embedding Number of rows (same for each embedding)
 *  @param nr_cols Embedding Number of columns (same for each embedding)
 *  @param nr_embedding number of embedding in emb_tables
 */
int32_t **
alloc_emb_tables(uint32_t nr_rows, uint32_t nr_cols, uint32_t nr_embedding) {
    int32_t **emb_tables = (int32_t **) malloc(nr_embedding * sizeof(int32_t *));
    for (uint32_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* allocate embedding table on host side */
        emb_tables[embedding_index] = (int32_t *) malloc(nr_rows * nr_cols * sizeof(int32_t));
    }
    return emb_tables;
}

void
free_emb_tables(int32_t **emb_tables, uint64_t nr_embedding) {
    for (uint32_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* free embedding table */
        free(emb_tables[embedding_index]);
    }
    free(emb_tables);
}

void
synthetic_populate(int32_t **emb_tables, uint32_t nr_rows, uint32_t nr_cols,
                   uint32_t nr_embedding) {

    for (uint32_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* synthetize embedding table parameters */
        for (int i = 0; i < nr_rows * nr_cols; i++) {
            double data_norm = (double) (rand()) / RAND_MAX / INDEX_PER_BATCH;
            emb_tables[embedding_index][i] = (int32_t) (INT32_MAX * data_norm);
            // table_data[i] = (int32_t) i;
        }
    }

    /* store one embedding to DPU MRAM */
    populate_mram(nr_embedding, nr_rows, nr_cols, emb_tables, NULL);
}

/** @brief check DPU embedding inference result for each embedding and each batch
 *  @param emb_tables host side embeding tables
 *  @param nr_embedding number of embedding in emb_tables
 *  @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
 *  @param offsets array that stores indices offset (pytorch EmbedingBag convention)
 *  [EMB_INDEX][BATCH_INDEX][OFFSET]
 *  @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
 *  @param nr_batches gives the number of batch (same for each embedding) in indices
 *  @param nr_cols Embedding Number of columns (same for each embedding)
 *  @param results DPU embedding inference result buffer [EMB_INDEX][BATCH_INDEX * NR_COLS]
 *  @return host model result and DPU results are the same or not
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
        /* for each input batch of index */
        for (int batch_index = 0; batch_index < nr_batches; batch_index++) {
            /* reset tmb buffer */
            for (int col_index = 0; col_index < nr_cols; col_index++)
                tmp_result[col_index] = 0;
            /* check limits */
            uint32_t upper_bound = batch_index == nr_batches - 1 ? indices_len[embedding_index]
                                                                 : offsets[embedding_index][batch_index + 1];
            for(uint32_t ind_ptr = offsets[embedding_index][batch_index]; ind_ptr < upper_bound; ind_ptr++) {
                /* solve ind_ptr */
                index = indices[embedding_index][ind_ptr];
                for (int col_index = 0; col_index < nr_cols; col_index++)
                    /*Embedding reduction mode : ADD */
                    tmp_result[col_index] +=
                        emb_tables[embedding_index][index * nr_cols + col_index];
            }
            /* ckeck the batch result */
            for (int col_index = 0; col_index < nr_cols; col_index++) {

                float dpu_result = results[embedding_index][batch_index * nr_cols + col_index];
                float host_result = tmp_result[col_index];
                // printf("%f %f \n", dpu_result* pow(10, 9)  , host_result);
                // printf("%f %f \n", dpu_result* pow(10, 9)  , host_result);
                float diff = fabs(dpu_result * pow(10, 9) - host_result);
                printf("[%d][%d][%d]diff %f\n", embedding_index, batch_index, col_index, diff);
                /* check magnitude with arbitrary threshold */
                if (fabs(dpu_result * pow(10, 9) - host_result) > 1000)
                    valid = false;
            }
        }
    }
    return valid;
}

/** @brief perform DPU embedding table inference given input indices with multiple embedding and
 * multiple batch
 *  @param final_results embedding lookup operation DPU results
 *  @param nr_embedding number of embedding in emb_tables
 *  @param nr_batches gives the number of batch (same for each embedding) in indices
 *  @param indices_pet_batch numbr of indices per batch
 *  @param nr_rows Embedding Number of rows (same for each embedding)
 *  @param nr_cols Embedding Number of columns (same for each embedding)
 */
void
synthetic_inference(int32_t **emb_tables, float **final_results, uint32_t nr_embedding,
                    uint32_t nr_batches, uint32_t indices_per_batch, uint32_t nr_rows,
                    uint32_t nr_cols) {

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

    struct timespec start, end;
    int sum = 0;
    for (int i = 0; i < NR_RUN; i++) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        lookup(indices, offsets, indices_len, nr_batches_per_embedding, final_results);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        sum += time_diff(start, end).tv_nsec;
    }
    __attribute__((unused)) bool valid;
    valid = check_embedding_set_inference(emb_tables, nr_embedding, indices, offsets, indices_len,
                                          nr_batches, nr_cols, final_results);
    // printf("inference : median latency [ms]: %lf, OK ? %d \n", 1e-6 * (double) sum / NR_RUN,
    //        (int) valid);
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
            final_results[k] = (float *) malloc(NR_BATCHES * NR_COLS * sizeof(uint32_t));
        }
    }

    NR_DPUS = NR_COLS * NR_EMBEDDING;

    printf("alloc dpus %lu \n", NR_DPUS);
    alloc_dpus(NR_DPUS);
    int32_t **emb_tables = alloc_emb_tables(NR_ROWS, NR_COLS, NR_EMBEDDING);
    synthetic_populate(emb_tables, NR_ROWS, NR_COLS, NR_EMBEDDING);
    synthetic_inference(emb_tables, final_results, NR_EMBEDDING, NR_BATCHES, INDEX_PER_BATCH,
                        NR_ROWS, NR_COLS);
    free(final_results);
    free_emb_tables(emb_tables, NR_EMBEDDING);
}
