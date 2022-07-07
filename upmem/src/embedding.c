#include "embedding.h"

#include "dpu.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#define DPU_BINARY "./build/embdpu"

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

/** @brief alloc dpu set with given number of dpus */
void
alloc_dpus(uint64_t nr_dpus) {
    DPU_ASSERT(dpu_alloc(nr_dpus, "nrJobsPerRank=256", &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
}

void
free_embedding_rank_mapping(embedding_rank_mapping *rank_mapping) {
    for (uint32_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++) {
        free(rank_mapping->embedding_index[rank_index]);
        free(rank_mapping->embedding_start_pos[rank_index]);
    }
    free(rank_mapping->embedding_index);
    free(rank_mapping->embedding_start_pos);
    free(rank_mapping->rank_nr_dpus);
    free(rank_mapping->rank_nr_embedding);
    free(rank_mapping);
}

/** @brief transfer one embedding table params to DPU DRAM
 *  @param populate_mram(uint32_t
 *  @param embedding_id index of the embedding table to transfer
 *  @param nr_rows embedding number of rows (common to all embedding)
 *  @param table_data stores multiple embedding parameters
 */
embedding_rank_mapping *
populate_mram(uint64_t nr_embedding, uint64_t nr_rows, uint64_t nr_cols, int32_t **emb_tables,
              dpu_runtime_totals *runtime) {

    /* solves rank embeding mappin */
    embedding_rank_mapping *rank_mapping = malloc(sizeof(embedding_rank_mapping));
    {
        DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &(rank_mapping->nr_ranks)));
        rank_mapping->embedding_index = malloc(rank_mapping->nr_ranks * sizeof(uint32_t *));
        rank_mapping->embedding_start_pos = malloc(rank_mapping->nr_ranks * sizeof(uint32_t *));
        rank_mapping->rank_nr_dpus = malloc(rank_mapping->nr_ranks * sizeof(uint32_t));
        rank_mapping->rank_nr_embedding = malloc(rank_mapping->nr_ranks * sizeof(uint32_t));
        struct dpu_set_t rank;
        uint32_t rank_index = 0;
        DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
            DPU_ASSERT(dpu_get_nr_dpus(rank, &(rank_mapping->rank_nr_dpus[rank_index])));
            printf("rank nr dpus %u \n", rank_mapping->rank_nr_dpus[rank_index]);
            assert(nr_cols <= rank_mapping->rank_nr_dpus[rank_index]);
        }

        uint32_t cur_col = 0;
        uint32_t cur_emb = 0;
        for (uint32_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++) {
            uint32_t rank_nr_full_embedding_ = rank_mapping->rank_nr_dpus[rank_index] / nr_cols;
            uint32_t rank_nr_partial_embedding_ = 0;
            if ((rank_mapping->rank_nr_dpus[rank_index] % nr_cols))
                rank_nr_partial_embedding_ += 1;
            rank_mapping->rank_nr_embedding[rank_index] =
                rank_nr_full_embedding_ + rank_nr_partial_embedding_;
            rank_mapping->embedding_index[rank_index] =
                malloc(rank_mapping->rank_nr_embedding[rank_index] * sizeof(uint32_t));
            rank_mapping->embedding_start_pos[rank_index] =
                malloc(rank_mapping->rank_nr_embedding[rank_index] * sizeof(uint32_t));
            uint32_t cur_emb_ = 0;
            for (; cur_emb_ < rank_nr_full_embedding_; cur_emb_++) {
                rank_mapping->embedding_index[rank_index][cur_emb_] = cur_emb;
                rank_mapping->embedding_start_pos[rank_index][cur_emb_] = cur_col;
                cur_col += nr_cols;
                cur_emb += 1;
            }
            if (rank_nr_partial_embedding_) {
                rank_mapping->embedding_index[rank_index][cur_emb_] = cur_emb;
                rank_mapping->embedding_start_pos[rank_index][cur_emb_] = cur_col;
                cur_col +=
                    (rank_mapping->rank_nr_dpus[rank_index] - rank_nr_full_embedding_ * nr_cols);
            }
        }
    }
    // for (uint32_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++) {
    //     printf("rank [%u] %u \n", rank_index, rank_mapping->rank_nr_embedding[rank_index]);
    //     for (uint32_t cur_emb_ = 0; cur_emb_ < rank_mapping->rank_nr_embedding[rank_index];
    //          cur_emb_++) {
    //         printf("rank [%u][%u] : index %u start %u \n", rank_index, cur_emb_,
    //                rank_mapping->embedding_index[rank_index][cur_emb_],
    //                rank_mapping->embedding_start_pos[rank_index][cur_emb_]);
    //     }
    // }
    uint32_t nr_dpus;
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));
    assert(nr_embedding * nr_cols == (uint64_t) nr_dpus);

    /* allocates ant creates transpose embeding matrix of parameters */
    int32_t **buffer_data;
    buffer_data = (int32_t **) (malloc(nr_embedding * sizeof(int32_t *)));
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {
        buffer_data[embedding_index] = (int32_t *) (malloc(nr_rows * nr_cols * sizeof(int32_t)));

        for (uint64_t row_index = 0; row_index < nr_rows; row_index++)
            for (uint64_t col_index = 0; col_index < nr_cols; col_index++) {
                buffer_data[embedding_index][col_index * nr_rows + row_index] =
                    emb_tables[embedding_index][row_index * nr_cols + col_index];
            }
    }

    struct dpu_set_t dpu;
    uint64_t embedding_index = 0;
    uint64_t cur_emb_cols = 0;
    DPU_FOREACH(dpu_set, dpu) {
        /* set start addr of each transposed column */
        uint64_t col_start_addr = cur_emb_cols * nr_rows;
        DPU_ASSERT(dpu_prepare_xfer(dpu, &(buffer_data[embedding_index][col_start_addr])));

        cur_emb_cols++;
        if (cur_emb_cols == nr_cols) {
            embedding_index++;
            cur_emb_cols = 0;
        }
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "emb_data", 0,
                             ALIGN(nr_rows * sizeof(int32_t), 8), DPU_XFER_DEFAULT));

    DPU_ASSERT(
        dpu_broadcast_to(dpu_set, "emb_nr_rows", 0, &nr_rows, sizeof(uint64_t), DPU_XFER_DEFAULT));
    /* free transposed matrix of parameters */
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++)
        free(buffer_data[embedding_index]);
    free(buffer_data);

    // DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    // dpu_sync(dpu_set);
    // {
    //     struct dpu_set_t dpu;
    //     DPU_FOREACH(dpu_set, dpu) {
    //         DPU_ASSERT(dpu_log_read(dpu, stdout));
    //     }
    //     fflush(stdout);
    // }
    return rank_mapping;
}
struct callback_input {
    float **result_buffer;
    uint64_t *nr_batches_per_embedding;
    int32_t ***dpu_results_buffer;
    uint64_t nr_cols;
    uint64_t nr_embedding;
    embedding_rank_mapping *rank_mapping_info;
};

struct callback_input *callback_data = NULL;

/** @brief host side post processing of DPU side embedding results
 *  @param dpu_rank pointer to rank dpu set
 *  @param rank_id index of the rank
 *  @param args rank callback generic args
 */
dpu_error_t
gather_rank_embedding_results(struct dpu_set_t rank, uint32_t rank_index, void *cb_arg) {
    struct callback_input *input = (struct callback_input *) cb_arg;
    int32_t ***dpu_results_buffer = input->dpu_results_buffer;
    float **result_buffer = input->result_buffer;
    uint64_t *nr_batches_per_embedding = input->nr_batches_per_embedding;
    uint64_t nr_cols = input->nr_cols;
    embedding_rank_mapping *rank_mapping = input->rank_mapping_info;

    uint64_t rank_nr_embedding = rank_mapping->rank_nr_embedding[rank_index];
    for (uint32_t cur_emb = 0; cur_emb < rank_nr_embedding; cur_emb++) {

        uint64_t embedding_index = rank_mapping->embedding_index[rank_index][cur_emb];
        uint64_t embeddding_start_pos = rank_mapping->embedding_start_pos[rank_index][cur_emb];
        for (uint64_t batch_index = 0; batch_index < nr_batches_per_embedding[embedding_index];
             batch_index++) {
            for (uint64_t col_index = 0; col_index < nr_cols; col_index++) {
                result_buffer[embedding_index][batch_index * nr_cols + embeddding_start_pos +
                                               col_index] =
                    (float) dpu_results_buffer[embedding_index][embeddding_start_pos + col_index]
                                              [batch_index] *
                    pow(10, -9);
            }
        }
    }
    return DPU_OK;
}

/** @brief perform DPU lookup operation in embedding set and for input indices of
 *        multiple batch
 *  @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
 *  @param offsets array that stores indices offset (pytorch EmbedingBag convention)
 *  [EMB_INDEX][BATCH_INDEX][OFFSET]
 *  @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
 *  @param nr_batches_per_embedding gives the number of batch (same for each embedding) in indices
 *  @param result_buffer embedding lookup operation DPU results
 *  @return TBC
 */
int32_t *
lookup(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
       embedding_rank_mapping *rank_mapping_info, uint64_t nr_embedding, uint64_t nr_cols,
       float **result_buffer, int32_t ***dpu_result_buffer) {

    uint64_t dpu_index;
    uint64_t embedding_id;
    struct dpu_set_t dpu;
    struct query_len lengths[nr_embedding];

    // TODO: loop over embeddings
    DPU_FOREACH(dpu_set, dpu, dpu_index) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, indices[(int) (dpu_index / nr_cols)]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_indices", 0,
                             ALIGN(input_info->indices_len[0] * sizeof(uint32_t), 8),
                             DPU_XFER_ASYNC));

    // TODO: loop over embeddings
    DPU_FOREACH(dpu_set, dpu, dpu_index) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, offsets[(int) (dpu_index / nr_cols)]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_offsets", 0,
                             ALIGN(input_info->nr_batches_per_embedding[0] * sizeof(uint32_t), 8),
                             DPU_XFER_ASYNC));

    DPU_FOREACH(dpu_set, dpu, dpu_index) {
        embedding_id = (int) (dpu_index / nr_cols);
        // TODO : this functions support same batch size for each embedding, but
        assert(input_info->nr_batches_per_embedding[embedding_id] ==
               input_info->nr_batches_per_embedding[0]);
        lengths[embedding_id].indices_len = input_info->indices_len[0];
        lengths[embedding_id].nr_batches = input_info->nr_batches_per_embedding[embedding_id];
        DPU_ASSERT(dpu_prepare_xfer(dpu, &lengths[embedding_id]));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_lengths", 0, sizeof(struct query_len),
                             DPU_XFER_ASYNC));

    DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));

    DPU_FOREACH(dpu_set, dpu, dpu_index) {
        embedding_id = dpu_index / nr_cols;
        uint64_t dpu_mod_index = dpu_index % nr_cols;
        assert(input_info->nr_batches_per_embedding[embedding_id] ==
               input_info->nr_batches_per_embedding[0]);
        DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_result_buffer[embedding_id][dpu_mod_index]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "results", 0,
                             ALIGN(sizeof(int32_t) * input_info->nr_batches_per_embedding[0], 8),
                             DPU_XFER_ASYNC));

    callback_data->result_buffer = result_buffer;
    callback_data->nr_batches_per_embedding = input_info->nr_batches_per_embedding;
    callback_data->dpu_results_buffer = dpu_result_buffer;
    callback_data->rank_mapping_info = rank_mapping_info;
    DPU_ASSERT(
        dpu_callback(dpu_set, gather_rank_embedding_results, callback_data, DPU_CALLBACK_ASYNC));
    // DPU_ASSERT(dpu_sync(dpu_set));
    // DPU_FOREACH(dpu_set, dpu, dpu_index) {
    //     // if(dpu_index==0)
    //     DPU_ASSERT(dpu_log_read(dpu, stdout));
    // }

    return 0;
}

void
alloc_embedding_dpu_backend() {
    assert(callback_data == NULL);
    callback_data = malloc(sizeof(struct callback_input));
}

void
free_embedding_dpu_backend() {
    free(callback_data);
}