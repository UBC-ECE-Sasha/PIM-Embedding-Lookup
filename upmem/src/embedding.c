#include "embedding.h"

#define DPU_BINARY "./build/embdpu"

/** @brief host side embedding table buffer */
int32_t *buffer_data[NR_COLS];

/** @brief global referene to dpu_set */
struct dpu_set_t dpu_set;

// static void
// copy_interval(dpu_runtime_interval *interval, struct timespec *const start,
//               struct timespec *const end) {
//     interval->start.tv_nsec = start->tv_nsec;
//     interval->start.tv_sec = start->tv_sec;
//     interval->stop.tv_nsec = end->tv_nsec;
//     interval->stop.tv_sec = end->tv_sec;
// }

static int
alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows) {

    for (int j = 0; j < NR_COLS; j++) {

        size_t sz = nr_rows * sizeof(int32_t);
        buffer_data[j] = (int32_t *) malloc(ALIGN(sz, 8));
        if (buffer_data[j] == NULL) {
            return ENOMEM;
        }

        for (int k = 0; k < nr_rows; k++) {
            buffer_data[j][k] = table_data[k * NR_COLS + j];
        }
    }
    return 0;
}

void
free_buffers() {
    for (int j = 0; j < NR_COLS; j++) {
        free(buffer_data[j]);
    }
}

/** @brief alloc dpu set with given number of dpus */
void
alloc_dpus(uint64_t nr_dpus) {
    // assert(dpu_set == NULL);
    DPU_ASSERT(dpu_alloc(nr_dpus, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
}

/** @brief transfer one embedding table params to DPU DRAM
    @param populate_mram(uint32_t
    @param table_id index of the embedding table to transfer
    @param nr_rows embedding number of rows (common to all embedding)
    @param table_data stores multiple embedding parameters
*/
void
populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data,
              dpu_runtime_totals *runtime) {

    if (table_id >= AVAILABLE_RANKS) {
        fprintf(stderr, "%d ranks available but tried to load table %dth", AVAILABLE_RANKS,
                table_id);
        exit(1);
    }

    assert(alloc_buffers(table_id, table_data, nr_rows) == 0);

    struct dpu_set_t dpu;
    uint8_t dpu_id;

    DPU_FOREACH(dpu_set, dpu, dpu_id) {
        if (dpu_id < (table_id + 1) * NR_COLS && dpu_id > table_id * NR_COLS) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id - (table_id * NR_COLS)]));
        }
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "emb_data", 0,
                             ALIGN(nr_rows * sizeof(int32_t), 8), DPU_XFER_DEFAULT));

    free_buffers();
}

/** @brief host side post processing of DPU side embedding results
    @param dpu_rank pointer to rank dpu set
    @param rank_id index of the rank
    @param args rank callback generic args
*/
dpu_error_t
post_process(struct dpu_set_t dpu_rank, uint32_t rank_id, void *arg) {
    struct callback_input *input = (struct callback_input *) arg;
    float **final_results = input->final_results;
    uint32_t *nr_batches = input->nr_batches;
    dpu_error_t status = DPU_OK;
    if (rank_id < NR_TABLES) {
        for (int j = 0; j < NR_COLS; j++) {
            for (int k = 0; k < nr_batches[rank_id]; k++)
                final_results[rank_id][k * NR_COLS + j] =
                    (float) input->tmp_results[rank_id][j][k] * pow(10, -9);
        }
    }
    return status;
}

/** @brief perform DPU lookup operation in embedding set and for input indices of
        multiple batch
    @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
    @param offsets array that stores indices offset (pytorch EmbedingBag convention)
   [EMB_INDEX][BATCH_INDEX][OFFSET]
    @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
    @param nr_batches gives the number of batch (same for each embedding) in indices
    @param final_results embedding lookup operation DPU results
    @return TBC
*/
int32_t *
lookup(uint32_t **indices, uint32_t **offsets, uint32_t *indices_len, uint32_t *nr_batches,
       float **final_results) {
    int dpu_id;
    int table_id = 0;
    struct dpu_set_t dpu;
    struct query_len lengths[NR_TABLES];

    DPU_FOREACH(dpu_set, dpu, dpu_id) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, indices[(int) (dpu_id / NR_COLS)]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_indices", 0,
                             ALIGN(indices_len[0] * sizeof(uint32_t), 8), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, dpu_id) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, offsets[(int) (dpu_id / NR_COLS)]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_offsets", 0,
                             ALIGN(nr_batches[0] * sizeof(uint32_t), 8), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, dpu_id) {
        table_id = (int) (dpu_id / NR_COLS);
        lengths[table_id].indices_len = *indices_len;
        lengths[table_id].nr_batches = *nr_batches;
        DPU_ASSERT(dpu_prepare_xfer(dpu, &lengths[table_id]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_lengths", 0, sizeof(struct query_len),
                             DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));

    int32_t ***tmp_results = (int32_t ***) malloc(NR_TABLES * sizeof(int32_t **));
    DPU_FOREACH(dpu_set, dpu, dpu_id) {
        if (dpu_id % NR_COLS == 0) {
            table_id = dpu_id / NR_COLS;
            tmp_results[table_id] = (int32_t **) malloc(NR_COLS * sizeof(int32_t *));
        }
        tmp_results[table_id][dpu_id % NR_COLS] =
            (int32_t *) malloc(nr_batches[0] * sizeof(int32_t));
        DPU_ASSERT(dpu_prepare_xfer(dpu, &tmp_results[table_id][dpu_id % NR_COLS][0]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "results", 0,
                             ALIGN(sizeof(int32_t) * nr_batches[0], 8), DPU_XFER_DEFAULT));

    struct callback_input callback_data;
    callback_data.final_results = final_results;
    callback_data.nr_batches = nr_batches;
    callback_data.tmp_results = tmp_results;

    DPU_ASSERT(dpu_callback(dpu_set, post_process, (void *) &callback_data, DPU_CALLBACK_ASYNC));
    DPU_ASSERT(dpu_sync(dpu_set));

    /* if (runtime_group && RT_CONFIG == RT_LAUNCH) {
        if(runtime_group[table_id].in_use >= runtime_group[table_id].length) {
            TIME_NOW(&end);
            fprintf(stderr,
                "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                dpu_id, runtime_group[table_id].in_use, table_id, runtime_group[table_id].length);
            exit(1);
        }
        copy_interval(
            &runtime_group->intervals[runtime_group[table_id].in_use], &start, &end);
            runtime_group[table_id].in_use++;
    } */
    return 0;
}
