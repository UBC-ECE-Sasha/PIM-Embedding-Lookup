// To build the code: dpu-upmem-dpurte-clang -DNR_TASKLETS=16 -o dpu_multi dpu_multi.c
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mutex.h>
#include <stdbool.h> 

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t index_len_input;

__dma_aligned uint64_t read_buf[256];

int rows_per_tasklet;
int num_with_one_more;

// rows_per_tasklet+1, ..., rows_per_tasklet+1, rows_per_tasklet, ..., rows_per_tasklet
// rows_per_tasklet+1 will be repeaterd num_with_one_more times

uint64_t nr_rows, nr_cols, index_len;
bool init=false;

int main() {

    switch(me()){
        
        case 0:{

            nr_rows = row_size_input;
            nr_cols = col_size_input;
            index_len = index_len_input;

            rows_per_tasklet = index_len/10;
            num_with_one_more = index_len%10;

            __mram_ptr __dma_aligned uint8_t *mram_offset = DPU_MRAM_HEAP_POINTER;

            mram_offset += nr_rows*nr_cols*sizeof(uint64_t);

            //updating the contents of read_buf with the index of the rows that we will lookup
            mram_read(mram_offset, read_buf, index_len*sizeof(uint64_t));

            init=true;

            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }
        

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);
                    
                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;


        case 1:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 2:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 3:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 4:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 5:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 6:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 7:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 8:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 9:{

            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }


            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

         case 10:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 11:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 12:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 13:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 14:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;

        case 15:{
    
            while(!init){}

            __mram_ptr __dma_aligned uint8_t *mram_offset;
            uint64_t write_buf[16];
            int cur_index=0;
            int i=0;
            uint64_t read_len=nr_cols*sizeof(uint64_t);

            while(i<num_with_one_more && i<me()){
                cur_index += index_len+1;
                i++;
            }

            while(i<me()){
                cur_index += rows_per_tasklet;
                i++;
            }

            if(i<num_with_one_more){
                for(int j=0; j<rows_per_tasklet+1; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }

            else{
                for(int j=0; j<rows_per_tasklet; j++){

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                    mram_read(mram_offset, write_buf, read_len);

                    mram_offset = DPU_MRAM_HEAP_POINTER;
                    mram_offset += (nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf, mram_offset, read_len);

                    cur_index++;
                }
            }
        }
            break;
    }

    return 0;
}
