#include <mram.h>
#include <stdbool.h>
#include <stdint.h>

// To build the code: dpu-upmem-dpurte-clang -o emb_dpu emb_dpu.c

__mram_noinit uint8_t row_size_input;
__mram_noinit uint8_t col_size_input;
__mram_noinit uint8_t row_to_pool;
uint8_t __mram_ptr * emb;
uint8_t __mram_ptr * pooled_row;

int main() {

    uint8_t n, m, q;

    n = row_size_input;
    m = col_size_input;
    q = row_to_pool;

    emb = DPU_MRAM_HEAP_POINTER;
    
    for(unsigned int i=0; i< (unsigned int)m; i++){
        emb[n*m + i] = emb[q*m + i];
    }

    return 0;
}