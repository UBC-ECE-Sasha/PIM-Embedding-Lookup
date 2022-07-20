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
alloc_emb_tables(embedding_info *emb_info) {
    int32_t **emb_tables = (int32_t **) malloc(emb_info->nr_embedding * sizeof(int32_t *));
    for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
         embedding_index++) {
        /* allocate embedding table on host side */
        emb_tables[embedding_index] =
            (int32_t *) malloc(emb_info->nr_rows * emb_info->nr_cols * sizeof(int32_t));
    }
    return emb_tables;
}

void
free_emb_tables(int32_t **emb_tables, embedding_info *emb_info) {
    for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
         embedding_index++) {
        /* free embedding table */
        free(emb_tables[embedding_index]);
    }
    free(emb_tables);
}

enum emb_datagen { RAND, CPT, ZERO, NONE };
void
synthetic_populate(embedding_rank_mapping *rank_mapping, embedding_info *emb_info,
                   int32_t **emb_tables) {

    uint64_t emb_data_type = CPT;

    printf("generate synthetic tables\n");
    if (emb_data_type == RAND) {
        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            /* synthetize embedding table parameters */
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                double data_norm = (double) (rand()) / ((double) RAND_MAX + 1) / INDEX_PER_BATCH;
                emb_tables[embedding_index][i] = (int32_t) (INT32_MAX * data_norm);
            }
        }
    } else if (emb_data_type == CPT) {

        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            /* synthetize embedding table parameters */
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                emb_tables[embedding_index][i] = (int32_t) (embedding_index + i);
            }
        }
    } else if (emb_data_type == ZERO) {
        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            /* synthetize embedding table parameters */
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                emb_tables[embedding_index][i] = (int32_t) (0);
            }
        }
    }

    printf("populate mram with embedding synthetic tables\n");
    /* store one embedding to DPU MRAM */
    populate_mram(rank_mapping, emb_info, emb_tables);
}

void
check_one_embedding_set_inference(uint64_t embedding_index_, uint64_t *emb_valid,
                                  int32_t **emb_tables, uint32_t **indices, uint32_t **offsets,
                                  uint64_t *indices_len, uint64_t nr_batches, uint64_t nr_cols,
                                  float **results, uint64_t nr_embedding) {

    int32_t tmp_result[nr_cols];
    uint64_t index = 0;

    for (uint64_t embedding_index = embedding_index_;
         embedding_index < nr_embedding + embedding_index_; embedding_index++) {
        bool valid = true;
        /* for each embedding */
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
                for (int col_index = 0; col_index < nr_cols; col_index++) {
                    /*Embedding reduction mode : ADD */
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
                if (diff > 1000)
                    printf("[%lu][%d][%d] diff: %f\tdpu_result: %f\thost_result: %f\n",
                           embedding_index, batch_index, col_index, diff, dpu_result * pow(10, 9),
                           host_result);
                /* check magnitude with arbitrary threshold */
                if (diff > 1000)
                    valid = false;
            }
        }
        emb_valid[embedding_index] = valid;
    }
}

typedef struct {
    uint64_t embdedding_index;
    uint64_t *emb_valid;
    int32_t **emb_tables;
    uint32_t **indices;
    uint32_t **offsets;
    uint64_t *indices_len;
    uint64_t nr_batches;
    uint64_t nr_cols;
    float **results;
    uint64_t nr_embedding;
} param_map_t;

void *
thread_map_emb_check(void *argv) {
    param_map_t *param = (param_map_t *) argv;
    check_one_embedding_set_inference(param->embdedding_index, param->emb_valid, param->emb_tables,
                                      param->indices, param->offsets, param->indices_len,
                                      param->nr_batches, param->nr_cols, param->results,
                                      param->nr_embedding);

    pthread_exit(NULL);
}
pthread_t THREAD_MAP[MAX_NR_EMBEDDING];
#define CPU_NR_THREAD_MAX 100

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
check_embedding_set_inference(uint64_t nr_thread, int32_t **emb_tables, uint64_t nr_embedding,
                              uint32_t **indices, uint32_t **offsets, uint64_t *indices_len,
                              uint64_t nr_batches, uint64_t nr_cols, float **results) {

    uint64_t *emb_valid = malloc(nr_embedding * sizeof(uint64_t));
#define CPU_MT 1
#if (CPU_MT == 1)
    param_map_t **param = malloc(sizeof(param_map_t *) * nr_embedding);
    uint64_t emb_lot = nr_embedding / nr_thread;
    for (uint64_t embdedding_index = 0; embdedding_index < nr_embedding; embdedding_index++)
        param[embdedding_index] = (param_map_t *) malloc(sizeof(param_map_t));

    for (uint64_t embdedding_index = 0, emb_lot_index = 0; emb_lot_index < nr_thread;
         embdedding_index += emb_lot, emb_lot_index++) {

        param[emb_lot_index]->emb_valid = emb_valid;
        param[emb_lot_index]->embdedding_index = embdedding_index;
        param[emb_lot_index]->emb_tables = emb_tables;
        param[emb_lot_index]->indices = indices;
        param[emb_lot_index]->offsets = offsets;
        param[emb_lot_index]->indices_len = indices_len;
        param[emb_lot_index]->nr_batches = nr_batches;
        param[emb_lot_index]->nr_cols = nr_cols;
        param[emb_lot_index]->results = results;
        param[emb_lot_index]->nr_embedding = emb_lot;

        pthread_create(&THREAD_MAP[emb_lot_index], NULL, thread_map_emb_check,
                       param[emb_lot_index]);
    }

    for (int i = 0; i < nr_thread; i++)
        pthread_join(THREAD_MAP[i], NULL);

    for (int i = 0; i < nr_embedding; i++)
        free(param[i]);

    free(param);
#else
    param_map_t **param = malloc(sizeof(param_map_t *) * 1);

    param[0] = (param_map_t *) malloc(sizeof(param_map_t));
    param[0]->emb_valid = emb_valid;
    param[0]->embdedding_index = 0;
    param[0]->emb_tables = emb_tables;
    param[0]->indices = indices;
    param[0]->offsets = offsets;
    param[0]->indices_len = indices_len;
    param[0]->nr_batches = nr_batches;
    param[0]->nr_cols = nr_cols;
    param[0]->results = results;
    param[0]->nr_embedding = nr_embedding;

    pthread_create(&THREAD_MAP[0], NULL, thread_map_emb_check, param[0]);
    pthread_join(THREAD_MAP[0], NULL);

    for (int i = 0; i < 1; i++)
        free(param[i]);
    free(param);

#endif
    bool valid = true;
    for (uint64_t embdedding_index = 0; embdedding_index < nr_embedding; embdedding_index++) {
        // printf("vazlid %u\n", emb_valid[embdedding_index]);
        valid = valid && emb_valid[embdedding_index];
    }
    free(emb_valid);

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
alloc_offset_buffer(uint64_t nr_embedding, uint64_t nr_batches) {
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

input_info *
alloc_input_info(uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_indexes) {

    struct input_info *info = malloc(sizeof(struct input_info));
    info->nr_indexes = nr_indexes;
    info->indices_len = (uint64_t *) malloc(MAX_NR_EMBEDDING * sizeof(uint64_t));
    info->nr_batches = nr_batches;

    return info;
}
void
free_input_info(struct input_info *info) {
    free(info->indices_len);
}

embedding_info *
alloc_embedding_info(uint32_t nr_embedding, uint32_t nr_rows, uint32_t nr_cols, uint32_t sizeT) {
    embedding_info *emb_info = malloc(sizeof(embedding_info));
    emb_info->nr_embedding = nr_embedding;
    emb_info->nr_rows = nr_rows;
    emb_info->nr_cols = nr_cols;
    emb_info->sizeT = sizeT;
    return emb_info;
};
void
free_embedding_info(embedding_info *emb_info) {
    free(emb_info);
}

void
build_synthetic_input_size(struct input_info *input_info, uint32_t **indices_per_batch,
                           uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_rows) {
    uint32_t index_per_batch;
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {

        input_info->nr_batches = nr_batches;
        input_info->indices_len[embedding_index] = 0;
        for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++) {
#if (RAND_INPUT_SIZE == 1)
            double index_per_batch_norm = ((double) rand() / RAND_MAX);
            index_per_batch = (uint32_t) (index_per_batch_norm * MAX_INDEX_PER_BATCH_RAND);
#else
            index_per_batch = INDEX_PER_BATCH;
#endif
            indices_per_batch[embedding_index][batch_index] = index_per_batch;
            input_info->indices_len[embedding_index] += index_per_batch;
        }
    }
}

void
build_synthetic_input_data(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
                           uint64_t nr_embedding, uint64_t nr_batches, uint32_t **indices_per_batch,
                           uint64_t nr_rows, uint64_t nr_cols) {

    for (uint64_t k = 0; k < nr_embedding; k++) {
        input_info->nr_batches = nr_batches;
        input_info->indices_len[k] = 0;
        for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++)
            input_info->indices_len[k] += indices_per_batch[k][batch_index];

        offsets[k][0] = 0;
        for (uint64_t batch_index = 1; batch_index < nr_batches; batch_index++)
            offsets[k][batch_index] =
                offsets[k][batch_index - 1] + indices_per_batch[k][batch_index - 1];

        for (uint64_t batch_index = 0; batch_index < input_info->nr_batches; batch_index++) {
            for (uint64_t j = 0; j < indices_per_batch[k][batch_index]; j++) {
                double index_norm = ((double) rand() / RAND_MAX);
                uint64_t index = (uint64_t) (nr_rows * index_norm);
                indices[k][offsets[k][batch_index] + j] = index;
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
synthetic_inference(uint32_t **indices, uint32_t **offsets, input_info *input_info,
                    embedding_rank_mapping *rank_mapping_info, int32_t **emb_tables,
                    float **result_buffer, int32_t **dpu_result_buffer, embedding_info *emb_info) {

    uint64_t nr_embedding = emb_info->nr_embedding;
    uint64_t nr_batches = input_info->nr_batches;
    uint64_t nr_rows = emb_info->nr_rows;
    uint64_t nr_cols = emb_info->nr_cols;
    double cpu_p_ratio = 0, dpu_p_ratio = 0;
    __attribute__((unused)) bool valid;
    double cpu_time = 0, dpu_time = 0;
    uint64_t multi_run = 20;
    for (int i = 0; i < multi_run; i++) {

        struct timespec start_time, start_process_time, stop_time, stop_process_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);

        /* inference */
        lookup(indices, offsets, input_info, rank_mapping_info, nr_embedding, nr_cols, nr_rows,
               result_buffer, dpu_result_buffer);
        clock_gettime(CLOCK_MONOTONIC_RAW, &stop_time);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_process_time);
        double time = (float) ((stop_time.tv_sec - start_time.tv_sec) * 1e9 + stop_time.tv_nsec -
                               start_time.tv_nsec) /
                      (1e6);

        double process_time =
            (float) ((stop_process_time.tv_sec - start_process_time.tv_sec) * 1e9 +
                     stop_process_time.tv_nsec - start_process_time.tv_nsec) /
            (1e6);

        printf("[PERF] DPU CLOCK RAW time [ms]: %5.2f\n"
               "[PERF] DPUTIME time [ms]: %5.2f\n"
               "[PERF] DPU PROCESS ratio: %5.2f\n",
               time, process_time, process_time / time * 100.0);

        dpu_p_ratio += process_time / time * 100.0;
        dpu_time += time;
    }
#if (CHECK_RESULTS == 1)
    {
        uint64_t cpu_nr_thread = CPU_NR_THREAD_MAX;
        if (cpu_nr_thread > nr_embedding)
            cpu_nr_thread = nr_embedding;
        else {
            while (nr_embedding % cpu_nr_thread)
                cpu_nr_thread--;
        }

        for (int i = 0; i < multi_run; i++) {
            struct timespec start_time, start_process_time, stop_time, stop_process_time;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);

            valid = check_embedding_set_inference(cpu_nr_thread, emb_tables, nr_embedding, indices,
                                                  offsets, input_info->indices_len, nr_batches,
                                                  nr_cols, result_buffer);
            clock_gettime(CLOCK_MONOTONIC_RAW, &stop_time);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_process_time);
            double time = (float) ((stop_time.tv_sec - start_time.tv_sec) * 1e9 +
                                   stop_time.tv_nsec - start_time.tv_nsec) /
                          (1e6);
            double process_time =
                (float) ((stop_process_time.tv_sec - start_process_time.tv_sec) * 1e9 +
                         stop_process_time.tv_nsec - start_process_time.tv_nsec) /
                (1e6);
            printf("[PERF] CPU CLOCK RAW time [ms]: %5.2f\n"
                   "[PERF] CPUTIME time [ms]: %5.2f\n"
                   "[PERF] CPU PROCESS ratio: %5.2f\n",
                   time, process_time, process_time / time * 100.0);

            cpu_p_ratio += process_time / time * 100.0;
            cpu_time += time;
        }
    }
#endif

    double dpu_time_ms = dpu_time / multi_run;
    double cpu_time_ms = cpu_time / multi_run;

#if (CHECK_RESULTS == 1)

    printf("dpu [ms]: %lf, cpu [ms] %lf, dpu acceleration %lf\n DPU PRATIO %f, CPU PRATIO %f, DPU "
           "OK ? %d \n",
           dpu_time_ms, cpu_time_ms, cpu_time_ms / dpu_time_ms, dpu_p_ratio / multi_run,
           cpu_p_ratio / multi_run, (int) valid);
#else
    printf("dpu [ms]: %lf\n", dpu_time_ms);
#endif
}

float **
alloc_result_buffer(embedding_info *emb_info, input_info *i_info) {
    float **result_buffer = (float **) malloc(emb_info->nr_embedding * sizeof(float *));
    for (uint64_t k = 0; k < emb_info->nr_embedding; k++) {
        result_buffer[k] =
            (float *) malloc(i_info->nr_batches * emb_info->nr_cols * sizeof(uint32_t));
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

int32_t **
alloc_dpu_result_buffer(embedding_rank_mapping *rank_mapping_info, embedding_info *emb_info,
                        input_info *i_info) {

    uint32_t nr_dpus = rank_mapping_info->nr_dpus;
    uint64_t nr_batches = i_info->nr_batches;
    uint64_t nr_cols_per_dpu = rank_mapping_info->nr_cols_per_dpu;

    int32_t **dpu_result_buffer = (int32_t **) malloc(nr_dpus * sizeof(int32_t *));
    for (uint64_t k = 0; k < nr_dpus; k++) {
        dpu_result_buffer[k] = (int32_t *) malloc(nr_batches * nr_cols_per_dpu * sizeof(int32_t));
    }
    return dpu_result_buffer;
}

void
free_dpu_result_buffer(uint32_t nr_dpus, int32_t **dpu_result_buffer) {
    for (uint64_t k = 0; k < nr_dpus; k++) {
        free(dpu_result_buffer[k]);
    }
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
alloc_fifo_pool(embedding_info *emb_info, input_info *i_info) {

    struct FIFO_POOL *this = malloc(sizeof(struct FIFO_POOL));

    {
        printf("alloc FIFO [build_synthetic_input_data->inference], DEPTH(%u)\n", STAGE_0_DEPTH);
        input_batch *batch = malloc(sizeof(input_batch) * STAGE_0_DEPTH);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            uint32_t **indices = alloc_indices_buffer(emb_info->nr_embedding, i_info->nr_batches,
                                                      MAX_INDEX_PER_BATCH);
            uint32_t **offsets = alloc_offset_buffer(emb_info->nr_embedding, i_info->nr_batches);
            batch[i].indices = indices;
            batch[i].offsets = offsets;
            batch[i].input_info = i_info;
        }
        FIFO_INIT(&(this->stage_0), (void *) (batch), STAGE_0_DEPTH, sizeof(input_batch), 1, 1);
    }

    return this;
}

void
free_fifo_pool(struct FIFO_POOL *this, embedding_info *emb_info) {

    {
        FIFO *fifo = &(this->stage_0);
        printf("free FIFO [build_synthetic_input_data->inference], DEPTH(%lu)\n", fifo->depth);
        input_batch *batch = (input_batch *) (fifo->items[0]);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            free_indices_buffer(batch[i].indices, emb_info->nr_embedding);
            free_offset_buffer(batch[i].offsets, emb_info->nr_embedding);
        }
        free(batch);
        FIFO_FREE(fifo);
    }
}
struct pipeline_args {
    embedding_rank_mapping *rank_mapping_info;
    embedding_info *emb_info;
    input_info *i_info;
};

/**
 * @brief TBC
 *
 * @param argv NULL
 */
void *
thread_build_sythetic_data(void *argv) {

    struct pipeline_args *args = (struct pipeline_args *) argv;

    embedding_info *emb_info = args->emb_info;
    input_info *i_info = args->i_info;

    uint64_t nr_rows = emb_info->nr_rows;
    uint64_t nr_cols = emb_info->nr_cols;
    uint64_t nr_embedding = emb_info->nr_embedding;
    uint64_t nr_batches = i_info->nr_batches;

    FIFO *OUTPUT_FIFO = &(FIFO_POOL->stage_0);
    uint64_t total_batch = 0;

    uint32_t **indices_per_batch;
    indices_per_batch = malloc(MAX_NR_EMBEDDING * sizeof(uint32_t *));
    printf("max nr embedding %u\n", MAX_NR_EMBEDDING);
    for (uint64_t batch_index = 0; batch_index < MAX_NR_EMBEDDING; batch_index++)
        indices_per_batch[batch_index] = malloc(MAX_NR_BATCHES * sizeof(uint32_t));

    while (1) {
        input_batch *batch = FIFO_PUSH_RESERVE(input_batch, *OUTPUT_FIFO);
        batch->valid = 1;

        if (!(total_batch++ < NR_RUN)) {
            batch->valid = 0;
            FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
            break;
        }

        /* creates synthetic input batch of data */
        build_synthetic_input_size(batch->input_info, indices_per_batch, nr_embedding, nr_batches,
                                   nr_rows);
        /* creates synthetic input batch of data */
        build_synthetic_input_data(batch->indices, batch->offsets, batch->input_info, nr_embedding,
                                   nr_batches, indices_per_batch, nr_rows, nr_cols);

        /* release input FIFO */
        FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
    }

    for (uint64_t batch_index = 0; batch_index < MAX_NR_BATCHES; batch_index++)
        free(indices_per_batch[batch_index]);
    free(indices_per_batch);

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
    return NULL;
}
/**
 * @brief INIT_THREAD_POOL : initialize thread pool
 * @param this thread pool pointer
 */
void
INIT_THREAD_POOL(struct THREAD_POOL *this, embedding_rank_mapping *rank_mapping_info,
                 embedding_info *emb_info, input_info *i_info) {
    struct pipeline_args *pargs = (struct pipeline_args *) malloc(sizeof(struct pipeline_args));
    pargs->rank_mapping_info = rank_mapping_info;
    pargs->emb_info = emb_info;
    pargs->i_info = i_info;
    /* index on current thread to of THREAD_POOL structure */
    uint64_t thread_pool_index = 0;
    pthread_create(&(this->th[thread_pool_index++]), NULL, thread_build_sythetic_data,
                   (void *) pargs);
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

    uint64_t nr_embedding = NR_EMBEDDING;
    uint64_t nr_batches = NR_BATCHES;
    uint64_t index_per_batch = INDEX_PER_BATCH;
    uint64_t nr_cols = NR_COLS;
    uint64_t nr_rows = NR_ROWS;

    input_info *i_info = alloc_input_info(nr_embedding, nr_batches, index_per_batch);
    embedding_info *emb_info =
        alloc_embedding_info(nr_embedding, nr_rows, nr_cols, sizeof(int32_t));

    FIFO_POOL = alloc_fifo_pool(emb_info, i_info);
    alloc_embedding_dpu_backend();

    /* alloc final results buffer */
    printf("map embeddings on DPUs\n");
    embedding_rank_mapping *rank_mapping = embedding_dpu_map(emb_info, i_info);

    printf("nr cols per dpu %lu\n", rank_mapping->nr_cols_per_dpu);

    uint32_t nr_dpus = rank_mapping->nr_dpus;
    alloc_dpus(nr_dpus);
    printf("alloc dpus %u\n", nr_dpus);
    int32_t **emb_tables = alloc_emb_tables(emb_info);

    float **result_buffer = alloc_result_buffer(emb_info, i_info);
    int32_t **dpu_result_buffer = alloc_dpu_result_buffer(rank_mapping, emb_info, i_info);

    /* creates synthetic embedding parametes and transfet it to DPU MRAM */

    synthetic_populate(rank_mapping, emb_info, emb_tables);

    INIT_THREAD_POOL(&THREAD_POOL, rank_mapping, emb_info, i_info);

    FIFO *INPUT_FIFO = &(FIFO_POOL->stage_0);
    printf("start inference\n");
    while (1) {

        /* get one mapped read batch */
        input_batch *batch = FIFO_POP_RESERVE(input_batch, *INPUT_FIFO);

        if (!batch->valid) {
            FIFO_POP_RELEASE(*INPUT_FIFO);
            break;
        }

        /* perform inference */
        synthetic_inference(batch->indices, batch->offsets, batch->input_info, rank_mapping,
                            emb_tables, result_buffer, dpu_result_buffer, emb_info);

        FIFO_POP_RELEASE(*INPUT_FIFO);
    }
    JOIN_THREAD_POOL(&THREAD_POOL);

    free_result_buffer(result_buffer, emb_info->nr_embedding);
    free_dpu_result_buffer(nr_dpus, dpu_result_buffer);
    free_emb_tables(emb_tables, emb_info);

    free_fifo_pool(FIFO_POOL, emb_info);
    free_embedding_dpu_backend();
    free_embedding_info(emb_info);
    free_input_info(i_info);
}
