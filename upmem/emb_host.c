#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifndef DPU_BINARY
#define DPU_BINARY "emb_dpu"
#endif

uint8_t nr_rows;
uint8_t nr_cols;

// to build the code: gcc -O3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags --libs dpu`

void populate_mram(struct dpu_set_t set) {

    uint8_t nr_rows = 4;
    uint8_t nr_cols = 5;
    uint8_t i = 2;

    uint8_t buffer[20];

    int byte_index;

    DPU_ASSERT(dpu_copy_to(set, "row_to_pool", 0, (const uint8_t *)&i, sizeof(i)));
    DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint8_t *)&nr_rows, sizeof(nr_rows)));
    DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint8_t *)&nr_cols, sizeof(nr_cols)));

    for (byte_index = 0; byte_index < 20; byte_index++) {
        buffer[byte_index] = (uint8_t)byte_index;
    }

    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, 0, buffer, 20));
}


int main() {
    struct dpu_set_t set, dpu;
    uint8_t ans[5];

    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    
    populate_mram(set);

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 20, ans, 5));
        printf("ans[0] = 0x%08x\n", ans[0]);
        printf("ans[1] = 0x%08x\n", ans[1]);
        printf("ans[2] = 0x%08x\n", ans[2]);
        printf("ans[3] = 0x%08x\n", ans[3]);
        printf("ans[4] = 0x%08x\n", ans[4]);

    }
    DPU_ASSERT(dpu_free(set));
    return 0;
}
