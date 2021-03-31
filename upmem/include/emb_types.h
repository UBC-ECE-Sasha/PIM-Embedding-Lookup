#include <stdint.h>
#include <stdbool.h>

#define MAX_ENC_BUFFER_SIZE MEGABYTE(MAX_ENC_BUFFER_MB)
#define MAX_CAPACITY MEGABYTE(14) //Must be a multiply of 2
#define DPUS_PER_RANK 64
#define AVAILABLE_RANKS 10
#define MAX_NR_BUFFERS 100

struct embedding_buffer {
    int32_t *data;
    uint64_t first_row, last_row;
    uint32_t table_id;
};

struct embedding_table {
    uint32_t first_dpu_id, last_dpu_id, nr_buffers;
    uint64_t nr_rows;
    struct embedding_buffer **buffers;
    int32_t *ans;
};

struct lookup_result {
    int32_t data[ALIGN(NR_COLS,2)];
    uint32_t id;
    bool is_complete;
};
