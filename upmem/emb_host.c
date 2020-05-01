#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <dpu_log.h>

#ifndef DPU_BINARY
#define DPU_BINARY "emb_dpu_lookup"
#endif


struct dpu_set_t set;

// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags --libs dpu`
// to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC emb_host.c `dpu-pkg-config --cflags --libs dpu`

void populate_mram(uint32_t dpu_n,uint8_t nr_rows, uint8_t nr_cols, float *data) {

    //uint32_t buffer[nr_rows][nr_cols];
    struct dpu_set_t dpu;
    uint8_t write_len=nr_cols*nr_rows*sizeof(float);
    if(write_len%8!=0)
        write_len+=8-(write_len%8);

    DPU_ASSERT(dpu_alloc(dpu_n, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    DPU_ASSERT(dpu_copy_to(set, "row_size_input", 0, (const uint8_t *)&nr_rows, sizeof(nr_rows)));
    DPU_ASSERT(dpu_copy_to(set, "col_size_input", 0, (const uint8_t *)&nr_cols, sizeof(nr_cols)));
    
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, 0, (const float *)data, write_len));

    printf("last data=%f\n",data[14]);
    return;
}

void lookup(float* input,uint8_t length, uint8_t nr_cols, uint8_t nr_rows){
    float ans[length*nr_cols];

    uint8_t write_len=length*sizeof(float);
    if(write_len%8!=0)
        write_len+=8-(write_len%8);

    DPU_ASSERT(dpu_copy_to(set, "index_len_input", 0, (const uint8_t *)&length, sizeof(length)));
    DPU_ASSERT(dpu_copy_to(set, DPU_MRAM_HEAP_POINTER_NAME, nr_cols*nr_rows*sizeof(float), (const float *)input, write_len));

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, (float)(nr_cols*nr_rows*sizeof(float)) , ans, sizeof(float)*nr_cols*length));
        DPU_ASSERT(dpu_log_read(dpu, stdout));
        //for (int i=0;i<length;i++){
            //for (int j=0; j<nr_cols;j++)
	           // printf("ans[%d][%d] = %f\n",(uint32_t)input[i],j, ans[i*nr_cols+j]);
	    //}
    }
    DPU_ASSERT(dpu_free(set));
}

int init_dpu(){
    //printf("size:%d\n",(unsigned int)sizeof(float));

    float data[15]={
        1.1,2.2,3.3,4.4,5.5,
        2.2,4.4,6.6,8.8,10.10,
        3.3,6.6,9.9,12.12,15.15,
    };
    float input[2]={1.0,2.0};
    uint8_t length=2;
    
    populate_mram(1,3,5,data);
    lookup(input,length,5,3);
    return 0;
}

int main(){
    init_dpu(1);
    return 0;
}