#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <dpu_log.h>

#ifndef DPU_BINARY
#define DPU_BINARY "emb_dpu_lookup"
#endif


struct dpu_set_t set;

// to build the code: gcc -O3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags --libs dpu`

void *populate_mram(uint32_t dpu_n,uint8_t nr_rows, uint8_t nr_cols, uint32_t *data) {

    //uint32_t buffer[nr_rows][nr_cols];
    struct dpu_set_t dpu;

    DPU_ASSERT(dpu_alloc(dpu_n, NULL, &set));
    DPU_ASSERT(dpu_load(set, "emb_dpu_lookup", NULL));

    DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint8_t *)&nr_rows, sizeof(nr_rows)));
    DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint8_t *)&nr_cols, sizeof(nr_cols)));
    
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, 0, (const uint32_t *)data, nr_cols*nr_rows*sizeof(uint32_t)));

    //printf("%d\n",data[0]);
    return (void *)&set;
}

void lookup(uint32_t* input,uint8_t length, uint8_t nr_cols, uint8_t nr_rows){
    uint32_t ans[length*nr_cols];

    DPU_ASSERT(dpu_copy_to(set, "index_len_input", 0, (const uint8_t *)&length, sizeof(length)));
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, nr_cols*nr_rows*sizeof(uint32_t), (const uint32_t *)input, length*sizeof(uint32_t)));

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, (uint32_t)(nr_cols*nr_rows*sizeof(uint32_t)) , ans, sizeof(uint32_t)*nr_cols*length));
        DPU_ASSERT(dpu_log_read(dpu, stdout));
        for (int i=0;i<length;i++){
            for (int j=0; j<nr_cols;j++)
	            printf("ans[%d][%d] = %d\n",input[i],j, ans[i*nr_cols+j]);
	    }
    }
    DPU_ASSERT(dpu_free(set));
}

int init_dpu(uint32_t dpu_n){
    struct dpu_set_t set;
    printf("n:%d\n",dpu_n);
    DPU_ASSERT(dpu_alloc(dpu_n, NULL, &set));
    DPU_ASSERT(dpu_load(set, "emb_dpu_lookup", NULL));

    uint32_t data[20]={
        1,2,3,4,5,
        2,4,6,8,10,
        3,6,9,12,15,
        4,8,12,16,20
    };
    uint32_t input[5]={0,1,2,3};
    uint8_t length=4;
    
    //populate_mram(set,4,5,data);
    //lookup(set,input,length,5,4);
    return 0;
}

int main(){
    return 0;
}
