#include "embedding.h"
#include "fifo.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t NR_DPUS;
struct FIFO_POOL *FIFO_POOL;

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
                __attribute__((unused)) float diff;
                diff = fabs(dpu_result * pow(10, 9) - host_result);
                // printf("[%d][%d][%d]diff: %f\tdpu_result: %f\thost_result: %f\n",
                // embedding_index,
                //       batch_index, col_index, diff, dpu_result * pow(10, 9), host_result);
                /* check magnitude with arbitrary threshold */
                if (fabs(dpu_result * pow(10, 9) - host_result) > 1000)
                    valid = false;
            }
        }
    }
    return valid;
}

uint32_t **
alloc_indices_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch) {
    uint32_t **indices = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        indices[k] = (uint32_t *) malloc(nr_batches * indices_per_batch * sizeof(uint32_t));
    }
    return indices;
}

void
free_indices_buffer(uint32_t **indices, uint64_t nr_embedding) {
    for (uint32_t k = 0; k < nr_embedding; k++) {
        free(indices[k]);
    }
    free(indices);
}

uint32_t **
alloc_offset_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch) {
    uint32_t **offsets = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        offsets[k] = (uint32_t *) malloc(nr_batches * sizeof(uint32_t));
    }
    return offsets;
}

void
free_offset_buffer(uint32_t **offsets, uint64_t nr_embedding) {
    for (uint32_t k = 0; k < nr_embedding; k++) {
        free(offsets[k]);
    }
    free(offsets);
}

struct input_info *
alloc_input_info(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch) {

    struct input_info *info = malloc(sizeof(struct input_info));
    info->indices_len = (uint64_t *) malloc(nr_embedding * sizeof(uint64_t));
    info->nr_batches_per_embedding = (uint64_t *) malloc(nr_embedding * sizeof(uint64_t));
    return info;
}
void
free_input_info(struct input_info *info) {
    free(info->indices_len);
    free(info->nr_batches_per_embedding);
}

void
build_synthetic_input_data(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
                           uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch,
                           uint64_t nr_rows, uint64_t nr_cols) {

    // creates synthetic input batch of indices
    for (uint64_t k = 0; k < nr_embedding; k++) {
        input_info->indices_len[k] = nr_batches * indices_per_batch;
        input_info->nr_batches_per_embedding[k] = nr_batches;
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
synthetic_inference(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
                    int32_t **emb_tables, float **result_buffer, int32_t ***dpu_result_buffer,
                    uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch,
                    uint64_t nr_rows, uint64_t nr_cols) {

    uint64_t multi_run = 1;
    struct timespec start, end;
    int sum = 0;

    for (int i = 0; i < multi_run; i++) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        lookup(indices, offsets, input_info, nr_embedding, nr_cols, result_buffer,
               dpu_result_buffer);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        sum += time_diff(start, end).tv_nsec;
    }
    __attribute__((unused)) bool valid;
    valid =
        check_embedding_set_inference(emb_tables, nr_embedding, indices, offsets,
                                      input_info->indices_len, nr_batches, nr_cols, result_buffer);

    printf("inference : median latency [ms]: %lf, OK ? %d \r", 1e-6 * (double) sum / NR_RUN,
           (int) valid);
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

int32_t ***
alloc_dpu_result_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_cols) {

    // TODO : reduncency with definition in synthetic_inference
    uint64_t *nr_batches_per_embedding = (uint64_t *) malloc(nr_embedding * sizeof(uint64_t));

    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        nr_batches_per_embedding[embedding_index] = nr_batches;
    }

    int32_t ***dpu_result_buffer = (int32_t ***) malloc(nr_embedding * sizeof(int32_t **));
    for (uint64_t dpu_index = 0; dpu_index < NR_DPUS; dpu_index++) {
        uint64_t embedding_index = dpu_index / nr_cols;
        uint64_t dpu_mod_index = dpu_index % nr_cols;
        if (dpu_mod_index == 0)
            dpu_result_buffer[embedding_index] = (int32_t **) malloc(nr_cols * sizeof(int32_t *));

        dpu_result_buffer[embedding_index][dpu_mod_index] =
            (int32_t *) malloc(nr_batches_per_embedding[embedding_index] * sizeof(int32_t));
    }

    // TODO remove this
    free(nr_batches_per_embedding);

    return dpu_result_buffer;
}

void
free_dpu_result_buffer(int32_t ***dpu_result_buffer, uint64_t nr_cols) {

    for (uint64_t dpu_index = 0; dpu_index < NR_DPUS; dpu_index++) {
        uint64_t embedding_index = dpu_index / nr_cols;
        uint64_t dpu_mod_index = dpu_index % nr_cols;
        free(dpu_result_buffer[embedding_index][dpu_mod_index]);
    }
    for (uint64_t dpu_index = 0; dpu_index < NR_DPUS; dpu_index++) {
        uint64_t embedding_index = dpu_index / nr_cols;
        uint64_t dpu_mod_index = dpu_index % nr_cols;
        if (dpu_mod_index == 0)
            free(dpu_result_buffer[embedding_index]);
    }
    // TODO remove this
    free(dpu_result_buffer);
}

#define THREAD_POOL_NR_THREAD 1
struct THREAD_POOL {
    pthread_t th[THREAD_POOL_NR_THREAD];
} THREAD_POOL;

#define STAGE_0_DEPTH 2
#define STAGE_1_DEPTH 2

struct FIFO_POOL {
    FIFO stage_0;
    FIFO stage_1;
};

struct FIFO_POOL *
alloc_fifo_pool() {

    struct FIFO_POOL *this = malloc(sizeof(struct FIFO_POOL));

    {
        printf("alloc FIFO [build_synthetic_input_data->inference], DEPTH(%u)\n", STAGE_0_DEPTH);
        input_batch *batch = malloc(sizeof(input_batch) * STAGE_0_DEPTH);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            struct input_info *input_info =
                alloc_input_info(NR_EMBEDDING, NR_BATCHES, INDEX_PER_BATCH);
            uint32_t **indices = alloc_indices_buffer(NR_EMBEDDING, NR_BATCHES, INDEX_PER_BATCH);
            uint32_t **offsets = alloc_offset_buffer(NR_EMBEDDING, NR_BATCHES, INDEX_PER_BATCH);
            batch[i].indices = indices;
            batch[i].offsets = offsets;
            batch[i].input_info = input_info;
        }
        FIFO_INIT(&(this->stage_0), (void *) (batch), STAGE_0_DEPTH, sizeof(input_batch), 1, 1);
    }
    {
        // printf("alloc FIFO [inference->merge_results], DEPTH(%u)\n", STAGE_1_DEPTH);
        // uint8_t *d_buff = malloc(PIPELINE_FIFO_COMMAND_DEPTH);
        // FIFO_INIT(&(this->command), (void *) (command_buff), PIPELINE_FIFO_COMMAND_DEPTH,
        //           sizeof(uint8_t), 1, 1);
    }
    return this;
}

void
free_fifo_pool(struct FIFO_POOL *this) {

    {
        FIFO *fifo = &(this->stage_0);
        printf("free FIFO [build_synthetic_input_data->inference], DEPTH(%lu)\n", fifo->depth);
        input_batch *batch = (input_batch *) (fifo->items[0]);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            free_indices_buffer(batch[i].indices, NR_EMBEDDING);
            free_offset_buffer(batch[i].offsets, NR_EMBEDDING);
            free_input_info(batch[i].input_info);
        }
        free(batch);
        FIFO_FREE(fifo);
    }
    {
        // FIFO *fifo = &(this->stage_1);
        // printf("free FIFO [inference->merge_results], DEPTH(%lu)\n", fifo->depth);
        // // uint8_t *buffer = (uint8_t *) (fifo->items[0]);
        // // free(buffer);
        // FIFO_FREE(fifo);
    }
}

/**
 * @brief TBC
 *
 * @param argv NULL
 */
void *
thread_build_sythetic_data(void *argv) {
    FIFO *OUTPUT_FIFO = &(FIFO_POOL->stage_0);
    uint64_t total_batch = 0;

    while (1) {
        input_batch *batch = FIFO_PUSH_RESERVE(input_batch, *OUTPUT_FIFO);
        batch->valid = 1;

        if (!(total_batch++ < NR_RUN)) {
            batch->valid = 0;
            FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
            break;
        }
        /* creates synthetic input batch of data */
        build_synthetic_input_data(batch->indices, batch->offsets, batch->input_info, NR_EMBEDDING,
                                   NR_BATCHES, INDEX_PER_BATCH, NR_ROWS, NR_COLS);

        /* release input FIFO */
        FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
    }

    /* thread exit */
    pthread_exit(NULL);
    return NULL;
}

/**
 * @brief TBC
 *
 * @param argv NULL
 */
void *
thread_mege_results(void *argv) {
    /* thread exit */
    // pthread_exit(NULL);
    return NULL;
}
/**
 * @brief INIT_THREAD_POOL : initialize thread pool
 * @param this thread pool pointer
 */
void
INIT_THREAD_POOL(struct THREAD_POOL *this) {
    /* index on current thread to of THREAD_POOL structure */
    uint64_t thread_pool_index = 0;
    pthread_create(&(this->th[thread_pool_index++]), NULL, thread_build_sythetic_data, NULL);
    //    pthread_create(&(this->th[thread_pool_index++]), NULL, thread_mege_results, NULL);
}

/**
 * @brief JOIN_THREAD_POOL : wait end of all thread in thread pool
 * @param this thread pool pointer
 */
void
JOIN_THREAD_POOL(struct THREAD_POOL *this) {
    for (uint64_t i = 0; i < THREAD_POOL_NR_THREAD; i++)
        pthread_join(this->th[i], NULL);
}

/** @brief synthetize embedding table, input indices and perform DPU embedding table */
int
main() {

    NR_DPUS = NR_COLS * NR_EMBEDDING;

    FIFO_POOL = alloc_fifo_pool();

    /* alloc final results buffer */
    float **result_buffer = alloc_result_buffer(NR_EMBEDDING, NR_BATCHES, NR_COLS);
    int32_t ***dpu_result_buffer = alloc_dpu_result_buffer(NR_EMBEDDING, NR_BATCHES, NR_COLS);

    printf("alloc dpus %lu \n", NR_DPUS);
    alloc_dpus(NR_DPUS);
    int32_t **emb_tables = alloc_emb_tables(NR_ROWS, NR_COLS, NR_EMBEDDING);

    /* creates synthetic embedding parametes and transfet it to DPU MRAM */
    synthetic_populate(emb_tables, NR_ROWS, NR_COLS, NR_EMBEDDING);

    INIT_THREAD_POOL(&THREAD_POOL);

    FIFO *INPUT_FIFO = &(FIFO_POOL->stage_0);
    while (1) {

        /* get one mapped read batch */
        input_batch *batch = FIFO_POP_RESERVE(input_batch, *INPUT_FIFO);

        if (!batch->valid) {
            FIFO_POP_RELEASE(*INPUT_FIFO);
            break;
        }

        /* perform inference */
        synthetic_inference(batch->indices, batch->offsets, batch->input_info, emb_tables,
                            result_buffer, dpu_result_buffer, NR_EMBEDDING, NR_BATCHES,
                            INDEX_PER_BATCH, NR_ROWS, NR_COLS);

        FIFO_POP_RELEASE(*INPUT_FIFO);
    }
    JOIN_THREAD_POOL(&THREAD_POOL);

    free_result_buffer(result_buffer, NR_EMBEDDING);
    free_dpu_result_buffer(dpu_result_buffer, NR_COLS);
    free_emb_tables(emb_tables, NR_EMBEDDING);

    free_fifo_pool(FIFO_POOL);
}