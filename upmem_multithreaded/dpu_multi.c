// To build the code: dpu-upmem-dpurte-clang -DNR_TASKLETS=16 -o dpu_multi dpu_multi.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mutex.h>

MUTEX_INIT(init_mutex);
MUTEX_INIT(my_mutex);

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t index_len_input;

bool init=false;
bool e[16] = { [0 ... 15 ] = false };

// x+1, x+1, ..., x+1, x, x, ..., x
// x+1 will be repeaterd a times
// x will be repeaterd 16-a times

int a, x; 

uint64_t nr_rows, nr_cols, index_len;

__dma_aligned uint64_t read_buf[256];
__dma_aligned uint64_t write_buf[16][16];
__dma_aligned uint64_t read_len, write_len;

__mram_ptr __dma_aligned uint8_t *mram_offset_read;
__mram_ptr __dma_aligned uint8_t *mram_offset_write;

int main() {

    mutex_lock(init_mutex);

    printf("#%d\n", me());

    if(!init){
        nr_rows = row_size_input;
        nr_cols = col_size_input;
        index_len = index_len_input;

        x = index_len/16;
        a = index_len%16;

        mram_offset_read = DPU_MRAM_HEAP_POINTER;

        read_len = index_len*sizeof(uint64_t);

        mram_offset_read += nr_rows*nr_cols*sizeof(uint64_t);

        //updating the contents of read_buf with the index of the rows that we will lookup
        mram_read(mram_offset_read, read_buf, read_len);

        init=true;
    }

    mutex_unlock(init_mutex);

    if(me()==0){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);
                
                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==1){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==2){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==3){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==4){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==5){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==6){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==7){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==8){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==9){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==10){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==11){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==12){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==13){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==14){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;
        
    }

    if(me()==15){

        while(!init){}

        int cur_index=0;

        int i=0;

        while(i<a && i<me()){
            cur_index += x+1;
            i++;
        }

        while(i<me()){
            cur_index += x;
            i++;
        }


        if(i<a){
            for(int j=0; j<x+1; j++){
                
                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        else{
            for(int j=0; j<x; j++){

                /*mutex_lock(my_mutex);

                printf("me():%d,  j:%d,  cur_index:%d,  read_buf[cur_index]:%lu\n", me(), j, cur_index, read_buf[cur_index]);

                mutex_unlock(my_mutex);*/

                mram_offset_read = DPU_MRAM_HEAP_POINTER+read_buf[cur_index]*nr_cols*sizeof(uint64_t);

                mram_offset_write = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len+cur_index*nr_cols)*sizeof(uint64_t);

                read_len = nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_read, write_buf[me()], read_len);

                mram_write(write_buf[me()], mram_offset_write, read_len);

                cur_index++;
            }
        }

        e[me()]=true;

        while(!e[0] || !e[1] || !e[2] || !e[3] || !e[4] || !e[5] || !e[6] || !e[7] || !e[8] || !e[9] || !e[10] || !e[11] || !e[12] || !e[13] || !e[14] || !e[15]){};

        mram_offset_read = DPU_MRAM_HEAP_POINTER+(nr_rows*nr_cols+index_len)*sizeof(uint64_t);

        mram_offset_write = DPU_MRAM_HEAP_POINTER+nr_rows*nr_cols*sizeof(uint64_t);

        write_len = nr_cols*sizeof(uint64_t);
        
        for(int j=0; j<index_len; j++){

            mram_read(mram_offset_read, write_buf[me()], write_len);

            mram_write(write_buf[me()], mram_offset_write, write_len);

            mram_offset_read += write_len;
            mram_offset_write += write_len;

        }
    }

    return 0;
}