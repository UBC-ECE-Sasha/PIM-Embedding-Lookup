#include "embedding.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t NR_DPUS;

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
alloc_emb_tables(uint64_t nr_rows, uint64_t nr_cols, uint64_t nr_embedding) {
    int32_t **emb_tables = (int32_t **) malloc(nr_embedding * sizeof(int32_t *));
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* allocate embedding table on host side */
        emb_tables[embedding_index] = (int32_t *) malloc(nr_rows * nr_cols * sizeof(int32_t));
    }
    return emb_tables;
}

void
free_emb_tables(int32_t **emb_tables, uint64_t nr_embedding) {
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* free embedding table */
        free(emb_tables[embedding_index]);
    }
    free(emb_tables);
}

void
synthetic_populate(int32_t **emb_tables, uint64_t nr_rows, uint64_t nr_cols,
                   uint64_t nr_embedding) {

    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* synthetize embedding table parameters */
        for (int i = 0; i < nr_rows * nr_cols; i++) {
            double data_norm = (double) (rand()) / RAND_MAX / INDEX_PER_BATCH;
            emb_tables[embedding_index][i] = (int32_t) (INT32_MAX * data_norm);
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
check_embedding_set_inference(int32_t **emb_tables, uint64_t nr_embedding, uint32_t **indices,
                              uint32_t **offsets, uint64_t *indices_len, uint64_t nr_batches,
                              uint64_t nr_cols, float **results) {
    bool valid = true;
    int32_t tmp_result[nr_cols];
    uint64_t index = 0;

    /* for each embedding */
    for (int embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        /* for each input batch of index */
        for (int batch_index = 0; batch_index < nr_batches; batch_index++) {
            /* reset tmb buffer */
            for (int col_index = 0; col_index < nr_cols; col_index++)
                tmp_result[col_index] = 0;
            /* check limits */
            uint64_t upper_bound = batch_index == nr_batches - 1 ?
                                       indices_len[embedding_index] :
                                       offsets[embedding_index][batch_index + 1];
            for (uint64_t ind_ptr = offsets[embedding_index][batch_index]; ind_ptr < upper_bound;
                 ind_ptr++) {
                /* solve ind_ptr */
                index = indices[embedding_index][ind_ptr];
                // assert(index < nr_batches * indices_len[embedding_index]);
                for (int col_index = 0; col_index < nr_cols;
                     col_index++) { /*Embedding reduction mode : ADD */
                    tmp_result[col_index] +=
                        emb_tables[embedding_index][index * nr_cols + col_index];
                }
            }
            /* ckeck the batch result */
            for (int col_index = 0; col_index < nr_cols; col_index++) {

                float dpu_result = results[embedding_index][batch_index * nr_cols + col_index];
                float host_result = tmp_result[col_index];
                // float diff = fabs(dpu_result * pow(10, 9) - host_result);
                // printf("[%d][%d][%d]diff: %f\tdpu_result: %f\thost_result: %f\n",
                // embedding_index, batch_index, col_index , diff, dpu_result * pow(10,9),
                // host_result);
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
 *  @param result_buffer embedding lookup operation DPU results
 *  @param nr_embedding number of embedding in emb_tables
 *  @param nr_batches gives the number of batch (same for each embedding) in indices
 *  @param indices_pet_batch numbr of indices per batch
 *  @param nr_rows Embedding Number of rows (same for each embedding)
 *  @param nr_cols Embedding Number of columns (same for each embedding)
 */
void
synthetic_inference(int32_t **emb_tables, float **result_buffer, uint64_t nr_embedding,
                    uint64_t nr_batches, uint64_t indices_per_batch, uint64_t nr_rows,
                    uint64_t nr_cols) {

    // allocate synthetic input/output batch of indices
    uint32_t **indices = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    uint32_t **offsets = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    uint64_t *indices_len = (uint64_t *) malloc(nr_embedding * sizeof(uint64_t));
    uint64_t *nr_batches_per_embedding = (uint64_t *) malloc(nr_embedding * sizeof(uint64_t));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        indices[k] = (uint32_t *) malloc(nr_batches * indices_per_batch * sizeof(uint32_t));
        offsets[k] = (uint32_t *) malloc(nr_batches * sizeof(uint32_t));
    }

    // creates synthetic input batch of indices
    for (uint64_t k = 0; k < nr_embedding; k++) {
        indices_len[k] = nr_batches * indices_per_batch;
        nr_batches_per_embedding[k] = nr_batches;
        for (uint64_t i = 0; i < nr_batches; i++) {
            offsets[k][i] = i * indices_per_batch;
            for (uint64_t j = 0; j < indices_per_batch; j++) {
                double index_norm = ((double) rand() / RAND_MAX);
                uint64_t index = (uint64_t) (nr_rows * index_norm);
                indices[k][i * indices_per_batch + j] = index;
                assert(index < nr_rows);
            }
        }
    }

    struct timespec start, end;
    double sum = 0;
    for (int i = 0; i < NR_RUN; i++) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        lookup(indices, offsets, indices_len, nr_batches_per_embedding, result_buffer);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        struct timespec diff = time_diff(start, end);
        sum += diff.tv_nsec + diff.tv_sec * 1000000000;
    }
    __attribute__((unused)) bool valid;
    valid = check_embedding_set_inference(emb_tables, nr_embedding, indices, offsets, indices_len,
                                          nr_batches, nr_cols, result_buffer);

    printf("inference : average latency [ms]: %lf, OK ? %d \n", 1e-6 * sum / NR_RUN,
           (int) valid);

    // free synthetic input/output batch
    for (uint64_t k = 0; k < nr_embedding; k++) {
        free(indices[k]);
        free(offsets[k]);
    }
    free(indices);
    free(offsets);
    free(indices_len);
    free(nr_batches_per_embedding);
}

float **
alloc_result_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_cols) {
    float **result_buffer = (float **) malloc(nr_embedding * sizeof(float *));
    for (uint64_t k = 0; k < nr_embedding; k++) {
        result_buffer[k] = (float *) malloc(nr_batches * nr_cols * sizeof(uint32_t));
    }
    return result_buffer;
}

void
free_result_buffer(float **buffer, uint64_t nr_embedding) {

    for (uint64_t k = 0; k < nr_embedding; k++) {
        free(buffer[k]);
    }
    free(buffer);
}

/** @brief synthetize embedding table, input indices and perform DPU embedding table */
int
main() {

    /* alloc final results buffer */
    float **result_buffer = alloc_result_buffer(NR_EMBEDDING, NR_BATCHES, NR_COLS);

    NR_DPUS = NR_COLS * NR_EMBEDDING;

    printf("alloc dpus %lu \n", NR_DPUS);
    alloc_dpus(NR_DPUS);
    int32_t **emb_tables = alloc_emb_tables(NR_ROWS, NR_COLS, NR_EMBEDDING);
    synthetic_populate(emb_tables, NR_ROWS, NR_COLS, NR_EMBEDDING);
    synthetic_inference(emb_tables, result_buffer, NR_EMBEDDING, NR_BATCHES, INDEX_PER_BATCH,
                        NR_ROWS, NR_COLS);

    free_result_buffer(result_buffer, NR_EMBEDDING);
    free_emb_tables(emb_tables, NR_EMBEDDING);
}
