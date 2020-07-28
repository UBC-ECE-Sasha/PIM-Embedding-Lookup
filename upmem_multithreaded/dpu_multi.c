// To build the code: dpu-upmem-dpurte-clang -DNR_TASKLETS=12 -o dpu_multi dpu_multi.c
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
__dma_aligned uint64_t write_buf[12][256];

__dma_aligned uint64_t cur_index[12];
__dma_aligned uint64_t base_index[12];
__dma_aligned uint64_t cur_num_rows[12];
__dma_aligned uint64_t read_len;

__mram_ptr __dma_aligned uint8_t *mram_offset_0;
__mram_ptr __dma_aligned uint8_t *mram_offset_1;
__mram_ptr __dma_aligned uint8_t *mram_offset_2;
__mram_ptr __dma_aligned uint8_t *mram_offset_3;
__mram_ptr __dma_aligned uint8_t *mram_offset_4;
__mram_ptr __dma_aligned uint8_t *mram_offset_5;
__mram_ptr __dma_aligned uint8_t *mram_offset_6;
__mram_ptr __dma_aligned uint8_t *mram_offset_7;
__mram_ptr __dma_aligned uint8_t *mram_offset_8;
__mram_ptr __dma_aligned uint8_t *mram_offset_9;
__mram_ptr __dma_aligned uint8_t *mram_offset_10;
__mram_ptr __dma_aligned uint8_t *mram_offset_11;

int rows_per_tasklet;
int num_with_one_more;

// rows_per_tasklet+1, ..., rows_per_tasklet+1, rows_per_tasklet, ..., rows_per_tasklet
// rows_per_tasklet+1 will be repeaterd num_with_one_more times

uint64_t nr_rows, nr_cols, index_len;
bool init=false;

//MUTEX_INIT(my_mutex);

int main() {

    switch(me()){
        
        case 0:{

            nr_rows = row_size_input;
            nr_cols = col_size_input;
            index_len = index_len_input;

            rows_per_tasklet = index_len/12;
            num_with_one_more = index_len%12;

            mram_offset_0 = DPU_MRAM_HEAP_POINTER;

            mram_offset_0 += nr_rows*nr_cols*sizeof(uint64_t);

            //updating the contents of read_buf with the index of the rows that we will lookup
            mram_read(mram_offset_0, read_buf, index_len*sizeof(uint64_t));

            for(int j=0; j<12; j++){
                cur_index[j]=0;
                base_index[j]=0;
                cur_num_rows[j]=0;
            }

            read_len = nr_cols*sizeof(uint64_t);

            init=true;

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }
        

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;

            int j=0; 

            while(j<till){
                
                mram_offset_0 = DPU_MRAM_HEAP_POINTER;
                mram_offset_0 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_0, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                
                j++;

                if((j*nr_cols)%256==0 || j==till){
                        mram_offset_0 = DPU_MRAM_HEAP_POINTER;
                        mram_offset_0 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                        mram_write(write_buf[me()], mram_offset_0, read_len*cur_num_rows[me()]);
                        
                        base_index[me()] = cur_index[me()];
                        cur_num_rows[me()] = 0;
                }
            }
        }
        break;


        case 1:{
    
            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                
                mram_offset_1 = DPU_MRAM_HEAP_POINTER;
                mram_offset_1 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_1, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                        mram_offset_1 = DPU_MRAM_HEAP_POINTER;
                        mram_offset_1 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                        mram_write(write_buf[me()], mram_offset_1, read_len*cur_num_rows[me()]);
                        
                        base_index[me()] = cur_index[me()];
                        cur_num_rows[me()] = 0;
                }
            }
        }
        break;

        case 2:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_2 = DPU_MRAM_HEAP_POINTER;
                mram_offset_2 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_2, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_2 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_2 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_2, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }         
        }
        break;

        case 3:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_3 = DPU_MRAM_HEAP_POINTER;
                mram_offset_3 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_3, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_3 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_3 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_3, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }       
        }
        break;

        case 4:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_4 = DPU_MRAM_HEAP_POINTER;
                mram_offset_4 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_4, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_4 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_4 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_4, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }       
        }
        break;

        case 5:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_5 = DPU_MRAM_HEAP_POINTER;
                mram_offset_5 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_5, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_5 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_5 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_5, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }        
        }
        break;

        case 6:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_6 = DPU_MRAM_HEAP_POINTER;
                mram_offset_6 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_6, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_6 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_6 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_6, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }       
        }
        break;

        case 7:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_7 = DPU_MRAM_HEAP_POINTER;
                mram_offset_7 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_7, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_7 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_7 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_7, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }      
        }
        break;

        case 8:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

            while(j<till){
                    
                mram_offset_8 = DPU_MRAM_HEAP_POINTER;
                mram_offset_8 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_8, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_8 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_8 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_8, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }            
        }
        break;

        case 9:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                int j=0; 

             while(j<till){
                    
                mram_offset_9 = DPU_MRAM_HEAP_POINTER;
                mram_offset_9 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_9, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_9 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_9 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_9, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }       
        }
        break;

        case 10:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_10 = DPU_MRAM_HEAP_POINTER;
                mram_offset_10 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_10, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_10 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_10 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_10, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }
        }
        break;

        case 11:{

            while(!init){}

            int i=0;
            int till;

            while(i<num_with_one_more && i<me()){
                cur_index[me()] += rows_per_tasklet+1;
                i++;
            }

            while(i<me()){
                cur_index[me()] += rows_per_tasklet;
                i++;
            }

            base_index[me()] = cur_index[me()];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_11 = DPU_MRAM_HEAP_POINTER;
                mram_offset_11 += read_buf[cur_index[me()]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_11, &write_buf[me()][(j%4)*nr_cols], read_len);

                cur_index[me()]++;
                cur_num_rows[me()]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_11 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_11 += (nr_rows*nr_cols+index_len+base_index[me()]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[me()], mram_offset_11, read_len*cur_num_rows[me()]);
                            
                    base_index[me()] = cur_index[me()];
                    cur_num_rows[me()] = 0;
                }
            }
        }
        break;
    }

    return 0;
}
