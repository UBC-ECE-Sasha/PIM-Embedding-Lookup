// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common/include/common.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef DPU_BINARY
#    define DPU_BINARY "../upmem/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

#define MAX_CAPACITY MEGABYTE(14) // Must be a multiply of 2
#define NR_TABLES 26
#define DPUS_PER_RANK 64
#define AVAILABLE_RANKS 10

struct embedding_buffer {
    int32_t *data;
    uint64_t first_row, last_row;
    uint64_t nr_rows, nr_cols;
    struct dpu_set_t *dpu;
    uint32_t table_id;
};

struct embedding_table {
    uint32_t first_dpu_id, last_dpu_id, nr_buffers;
    uint64_t nr_rows, nr_cols;
    struct embedding_buffer **buffers;
    int32_t *ans;
};

uint32_t total_buffers = 0, buff_arr_len = NR_TABLES;
uint32_t ready_buffers = 0, done_dpus = 0, allocated_ranks = 0;
struct embedding_buffer *buffers[MAX_ENC_BUFFER_SIZE];
struct embedding_table tables[NR_TABLES];
struct dpu_set_t *dpu_ranks[AVAILABLE_RANKS];

/*
    Params:
    0. table_id: embedding table number.
    1. nr_rows: number of rows of the embedding table
    2. nr_cols: number of columns of the embedding table
    3. table_data: a pointer of the size nr_rows*nr_cols containing table's data

    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
    corresponding table with the index of the first and last row held in each dpu.
*/
void
populate_mram(uint32_t table_id, uint64_t nr_rows, uint64_t nr_cols, int32_t *table_data) {
    uint64_t table_size = nr_rows * nr_cols;

    // first time calling populate_mram so allocate mem for buffers.
    if (table_id == 0) {
        buffers =
            (struct embedding_buffer **) malloc(NR_TABLES * sizeof(struct embedding_buffer *));
    }
    tables[table_id].nr_rows = nr_rows;
    tables[table_id].nr_cols = nr_cols;

    if (table_size <= MAX_CAPACITY) {
        tables[table_id].nr_buffers = 1;
        buffers[total_buffers] =
            (struct embedding_buffer *) malloc(sizeof(struct embedding_buffer));
        buffers[total_buffers]->first_row = 0;
        buffers[total_buffers]->last_row = nr_rows - 1;
        buffers[total_buffers]->nr_cols = nr_cols;
        buffers[total_buffers]->table_id = table_id;
    } else {
        tables[table_id].nr_buffers = (int) ((table_size * 1.0) / (MAX_CAPACITY * 1.0));
        if (table_size % MAX_CAPACITY != 0) {
            tables[table_id].nr_buffers += 1;
        }
        for (int j = 0; j < tables[table_id].nr_buffers; j++) {
            buff_arr_len++;
            buffers[total_buffers + j] =
                (struct embedding_buffer *) malloc(sizeof(struct embedding_buffer));
            buffers[total_buffers + j]->first_row = j * MAX_CAPACITY / nr_cols;
            buffers[total_buffers + j]->last_row =
                MIN(nr_rows - 1, ((j + 1) * MAX_CAPACITY) / nr_cols - 1);
            buffers = (struct embedding_buffer **) realloc(
                buffers, buff_arr_len * (sizeof(struct embedding_buffer *)));
            buffers[total_buffers + j]->table_id = table_id;
            buffers[total_buffers + j]->nr_cols = nr_cols;
        }
        tables[table_id].first_dpu_id = total_buffers;
        tables[table_id].last_dpu_id = total_buffers + tables[table_id].nr_buffers;
    }

    uint32_t first_row, last_row;

    for (int j = 0; j < tables[table_id].nr_buffers; j++) {
        first_row = buffers[total_buffers]->first_row;
        last_row = buffers[total_buffers]->last_row;
        buffers[total_buffers]->data =
            (int32_t *) malloc(ALIGN((last_row - first_row + 1) * nr_cols * sizeof(int32_t), 8));
        for (int k = 0; k < (last_row - first_row + 1) * nr_cols; k++) {
            buffers[total_buffers]->data[k] = table_data[(first_row * nr_cols) + k];
        }
        ready_buffers++;
        total_buffers++;
    }
    printf("done with %dth table\n", table_id);

    // Done with analyzing all tables or nr ready_buffers enough for a rank so
    // allocate a rank and copy embedding data.
    if (ready_buffers >= DPUS_PER_RANK || table_id == NR_TABLES - 1) {
        struct dpu_set_t set, dpu, dpu_rank;
        printf("allocating %d dpus and %d dpus allocated before.\n", ready_buffers, done_dpus);
        if (ready_buffers <= DPUS_PER_RANK)
            DPU_ASSERT(dpu_alloc(ready_buffers, NULL, &set));
        else
            DPU_ASSERT(dpu_alloc(DPUS_PER_RANK, NULL, &set));
        // TODO: make sure works correctly under this condition.
        DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

        uint32_t len;
        uint8_t dpu_id, rank_id;
        DPU_FOREACH(set, dpu, dpu_id) {
            first_row = buffers[done_dpus + dpu_id]->first_row;
            last_row = buffers[done_dpus + dpu_id]->last_row;
            buffers[done_dpus + dpu_id]->dpu = &dpu;
            len = (last_row - first_row + 1) * buffers[done_dpus + dpu_id]->nr_cols;

            // DPU_ASSERT(dpu_copy_to(dpu, "nr_rows_input", 0, (const uint64_t
            // *)&(buffers[done_dpus+dpu_id]->nr_rows), sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "nr_cols_input", 0,
                                   (const uint64_t *) &(buffers[done_dpus + dpu_id]->nr_cols),
                                   sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "first_row_input", 0, (const uint64_t *) &first_row,
                                   sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "last_row_input", 0, (const uint64_t *) &last_row,
                                   sizeof(uint64_t)));
            DPU_ASSERT(dpu_copy_to(dpu, "emb_data", 0,
                                   (const int32_t *) buffers[done_dpus + dpu_id]->data,
                                   ALIGN(len * sizeof(int32_t), 8)));
            printf("copied %dth buffer to dpu\n", dpu_id);

            buffers[done_dpus + dpu_id]->dpu = &dpu;
        }

        // This section is just for testing, see if correct values are in DPU
        /*int32_t ans[4];
        DPU_FOREACH(set, dpu, dpu_id){
            dpu_launch(dpu, DPU_SYNCHRONOUS);
            first_row=buffers[done_dpus+dpu_id]->first_row;
            last_row=buffers[done_dpus+dpu_id]->last_row;
            printf("In host last_row:%d, first row:%d &
        nr_cols:%d\n",last_row,first_row,buffers[done_dpus+dpu_id]->nr_cols); printf("first: %d,
        2nd: %d, -1: %d, last: %dfor %d buffer\n",buffers[dpu_id]->data[0],buffers[dpu_id]->data[1],
            buffers[dpu_id]->data[(last_row-first_row+1)*buffers[done_dpus+dpu_id]->nr_cols-2],
            buffers[dpu_id]->data[(last_row-first_row+1)*buffers[done_dpus+dpu_id]->nr_cols-1],dpu_id);
            uint32_t offset=
        ALIGN((last_row-first_row+1)*buffers[done_dpus+dpu_id]->nr_cols*sizeof(int32_t),8);
            printf("copying from dpu\n");
            DPU_ASSERT(dpu_copy_from(dpu, "ans_buffer", 0 , (int32_t*)ans, 2*sizeof(int32_t)));
            printf("copied from dpu\n");
            printf("%d: %d, %d\n",dpu_id,ans[0], ans[1]);
            printf("reading log\n");
            DPU_ASSERT(dpu_log_read(dpu, stdout));
            printf("log read for dpu %dth\n",dpu_id);
            printf("------------------------------------------\n");
            //printf("log printed");
        }*/

        for (int i = done_dpus; i < ready_buffers; i++)
            free(buffers[i]->data);

        // Assign dpus allocated to buffers to their embedding_tables.
        for (int i = 0; i < NR_TABLES; i++) {
            tables[i].buffers = (struct embedding_buffer **) malloc(
                tables[i].nr_buffers * sizeof(struct embedding_buffer *));
        }

        // dpu_launch(set, DPU_SYNCHRONOUS);

        uint32_t table_ptr = 0, tmp_ptr = 0;
        DPU_FOREACH(set, dpu, dpu_id) {
            if (tables[table_ptr].nr_buffers == tmp_ptr) {
                table_ptr++;
                tmp_ptr = 0;
            }
            tables[table_ptr].buffers[tmp_ptr] = buffers[dpu_id];

            // printf("------DPU %d Logs------\n", dpu_id);
            // DPU_ASSERT(dpu_log_read(dpu, stdout));
        }

        // done with a set of dpus, make changes to their counters to move to next set.
        if (ready_buffers <= DPUS_PER_RANK) {
            done_dpus += ready_buffers;
            ready_buffers = 0;
        } else {
            done_dpus += DPUS_PER_RANK;
            ready_buffers -= DPUS_PER_RANK;
        }
        dpu_ranks[allocated_ranks] = &set;
        allocated_ranks++;
    }
    return;
}

/*
    Params:
    1. ans: a pointer that be updated with the rows that we lookup
    2. input: a pointer containing the specific rows we want to lookup
    3. length: contains the number of rows that we want to lookup from the table
    4. nr_rows: number of rows of the embedding table
    5. nr_cols: number of columns of the embedding table

    Result:
    This function updates ans with the elements of the rows that we have lookedup
*/
int32_t *
lookup(uint32_t *indices, uint32_t *offsets, uint32_t *indices_len, uint32_t *offsets_len) {
    uint64_t max_len = 0;
    uint32_t index_sum = 0;
    for (int i = 0; i < NR_TABLES; i++) {
        for (int j = 0; j < tables[i].nr_buffers; j++) {
            printf("indices:%d\n", indices[index_sum]);
            sleep(10);
            DPU_ASSERT(dpu_prepare_xfer(*(tables[i].buffers[j]->dpu), &indices[index_sum]));
        }
        if (indices_len[i] > max_len)
            max_len = indices_len[i];
        index_sum += indices_len[i];
    }
    for (int k = 0; k < allocated_ranks; k++)
        DPU_ASSERT(dpu_push_xfer(*dpu_ranks[k], DPU_XFER_TO_DPU, "input_indices", 0,
                                 ALIGN(max_len, 8), DPU_XFER_DEFAULT));

    index_sum = 0;
    for (int i = 0; i < NR_TABLES; i++) {
        for (int j = 0; j < tables[i].nr_buffers; j++)
            DPU_ASSERT(dpu_prepare_xfer(*(tables[i].buffers[j]->dpu), &offsets[index_sum]));
        if (indices_len[i] > max_len)
            max_len = indices_len[i];
        index_sum += offsets_len[i];
    }
    for (int k = 0; k < allocated_ranks; k++)
        DPU_ASSERT(dpu_push_xfer(*dpu_ranks[k], DPU_XFER_TO_DPU, "input_offsets", 0,
                                 ALIGN(max_len, 8), DPU_XFER_DEFAULT));

    for (int i = 0; i < NR_TABLES; i++)
        for (int j = 0; j < tables[i].nr_buffers; j++)
            DPU_ASSERT(dpu_prepare_xfer(*(tables[i].buffers[j]->dpu), &indices_len[i]));
    for (int k = 0; k < allocated_ranks; k++)
        DPU_ASSERT(dpu_push_xfer(*dpu_ranks[k], DPU_XFER_TO_DPU, "input_indices_len", 0,
                                 sizeof(uint32_t), DPU_XFER_DEFAULT));

    for (int i = 0; i < NR_TABLES; i++)
        for (int j = 0; j < tables[i].nr_buffers; j++)
            DPU_ASSERT(dpu_prepare_xfer(*(tables[i].buffers[j]->dpu), &offsets_len[i]));
    for (int k = 0; k < allocated_ranks; k++)
        DPU_ASSERT(dpu_push_xfer(*dpu_ranks[k], DPU_XFER_TO_DPU, "input_offsets_len", 0,
                                 sizeof(uint32_t), DPU_XFER_DEFAULT));

    // for (int i=0; i<allocated_ranks; i++)
    // DPU_ASSERT(dpu_launch(dpu_ranks[i], DPU_ASYNCHRONOUS));

    return 0;
}

int
main() {

    // int32_t data[]={1,2,3,4,5,6,7,2,4,6,8,10,12,14,3,6,9,12,15,18,21,4,8,12,16,20,24,28};
    // populate_mram(0,4,7, data);
}
