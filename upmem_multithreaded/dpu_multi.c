// To build the code: dpu-upmem-dpurte-clang -DNR_TASKLETS=16 -o dpu_multi dpu_multi.c
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t index_len_input;

__dma_aligned uint64_t read_buf[256];
__dma_aligned uint64_t write_buf[16][256];

__dma_aligned uint64_t cur_index[16];
__dma_aligned uint64_t base_index[16];
__dma_aligned uint64_t cur_num_rows[16];
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
__mram_ptr __dma_aligned uint8_t *mram_offset_12;
__mram_ptr __dma_aligned uint8_t *mram_offset_13;
__mram_ptr __dma_aligned uint8_t *mram_offset_14;
__mram_ptr __dma_aligned uint8_t *mram_offset_15;

int rows_per_tasklet;
int num_with_one_more;
int answer_begin_index;

int cols_per_buf;

// rows_per_tasklet+1, ..., rows_per_tasklet+1, rows_per_tasklet, ..., rows_per_tasklet
// rows_per_tasklet+1 will be repeaterd num_with_one_more times

uint64_t nr_rows, nr_cols, index_len;

//MUTEX_INIT(my_mutex);
BARRIER_INIT(my_barrier, NR_TASKLETS);

int main() {

    switch(me()){
        
        case 0:{

            nr_rows = row_size_input;
            nr_cols = col_size_input;
            index_len = index_len_input;
            cols_per_buf = 256/nr_cols;

            rows_per_tasklet = index_len/16;
            num_with_one_more = index_len%16;

            answer_begin_index = nr_rows*nr_cols+index_len;

            mram_offset_0 = DPU_MRAM_HEAP_POINTER;
            mram_offset_0 += nr_rows*nr_cols*sizeof(uint64_t);

            //updating the contents of read_buf with the index of the rows that we will lookup
            mram_read(mram_offset_0, read_buf, index_len*sizeof(uint64_t));

            for(int j=0; j<16; j++){
                cur_index[j]=0;
                base_index[j]=0;
                cur_num_rows[j]=0;
            }

            read_len = nr_cols*sizeof(uint64_t);

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<0){
                cur_index[0] += rows_per_tasklet+1;
                i++;
            }
        

            while(i<0){
                cur_index[0] += rows_per_tasklet;
                i++;
            }

            base_index[0] = cur_index[0];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;

            int j=0; 

            while(j<till){
                
                mram_offset_0 = DPU_MRAM_HEAP_POINTER;
                mram_offset_0 += read_buf[cur_index[0]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_0, &write_buf[0][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[0]++;
                cur_num_rows[0]++;
                
                j++;

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_0 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_0 += (answer_begin_index+base_index[0]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[0], mram_offset_0, read_len*cur_num_rows[0]);
                    
                    base_index[0] = cur_index[0];
                    cur_num_rows[0] = 0;
                }
            }
        }
        break;


        case 1:{
    
            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<1){
                cur_index[1] += rows_per_tasklet+1;
                i++;
            }

            while(i<1){
                cur_index[1] += rows_per_tasklet;
                i++;
            }

            base_index[1] = cur_index[1];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                
                mram_offset_1 = DPU_MRAM_HEAP_POINTER;
                mram_offset_1 += read_buf[cur_index[1]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_1, &write_buf[1][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[1]++;
                cur_num_rows[1]++;
                
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_1 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_1 += (answer_begin_index+base_index[1]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[1], mram_offset_1, read_len*cur_num_rows[1]);
                    
                    base_index[1] = cur_index[1];
                    cur_num_rows[1] = 0;
                }
            }
        }
        break;

        case 2:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<2){
                cur_index[2] += rows_per_tasklet+1;
                i++;
            }

            while(i<2){
                cur_index[2] += rows_per_tasklet;
                i++;
            }

            base_index[2] = cur_index[2];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_2 = DPU_MRAM_HEAP_POINTER;
                mram_offset_2 += read_buf[cur_index[2]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_2, &write_buf[2][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[2]++;
                cur_num_rows[2]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_2 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_2 += (answer_begin_index+base_index[2]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[2], mram_offset_2, read_len*cur_num_rows[2]);
                            
                    base_index[2] = cur_index[2];
                    cur_num_rows[2] = 0;
                }
            }         
        }
        break;

        case 3:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<3){
                cur_index[3] += rows_per_tasklet+1;
                i++;
            }

            while(i<3){
                cur_index[3] += rows_per_tasklet;
                i++;
            }

            base_index[3] = cur_index[3];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_3 = DPU_MRAM_HEAP_POINTER;
                mram_offset_3 += read_buf[cur_index[3]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_3, &write_buf[3][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[3]++;
                cur_num_rows[3]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_3 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_3 += (answer_begin_index+base_index[3]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[3], mram_offset_3, read_len*cur_num_rows[3]);
                            
                    base_index[3] = cur_index[3];
                    cur_num_rows[3] = 0;
                }
            }       
        }
        break;

        case 4:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<4){
                cur_index[4] += rows_per_tasklet+1;
                i++;
            }

            while(i<4){
                cur_index[4] += rows_per_tasklet;
                i++;
            }

            base_index[4] = cur_index[4];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_4 = DPU_MRAM_HEAP_POINTER;
                mram_offset_4 += read_buf[cur_index[4]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_4, &write_buf[4][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[4]++;
                cur_num_rows[4]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_4 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_4 += (answer_begin_index+base_index[4]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[4], mram_offset_4, read_len*cur_num_rows[4]);
                            
                    base_index[4] = cur_index[4];
                    cur_num_rows[4] = 0;
                }
            }       
        }
        break;

        case 5:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<5){
                cur_index[5] += rows_per_tasklet+1;
                i++;
            }

            while(i<5){
                cur_index[5] += rows_per_tasklet;
                i++;
            }

            base_index[5] = cur_index[5];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_5 = DPU_MRAM_HEAP_POINTER;
                mram_offset_5 += read_buf[cur_index[5]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_5, &write_buf[5][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[5]++;
                cur_num_rows[5]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_5 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_5 += (answer_begin_index+base_index[5]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[5], mram_offset_5, read_len*cur_num_rows[5]);
                            
                    base_index[5] = cur_index[5];
                    cur_num_rows[5] = 0;
                }
            }        
        }
        break;

        case 6:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<6){
                cur_index[6] += rows_per_tasklet+1;
                i++;
            }

            while(i<6){
                cur_index[6] += rows_per_tasklet;
                i++;
            }

            base_index[6] = cur_index[6];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_6 = DPU_MRAM_HEAP_POINTER;
                mram_offset_6 += read_buf[cur_index[6]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_6, &write_buf[6][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[6]++;
                cur_num_rows[6]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_6 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_6 += (answer_begin_index+base_index[6]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[6], mram_offset_6, read_len*cur_num_rows[6]);
                            
                    base_index[6] = cur_index[6];
                    cur_num_rows[6] = 0;
                }
            }       
        }
        break;

        case 7:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<7){
                cur_index[7] += rows_per_tasklet+1;
                i++;
            }

            while(i<7){
                cur_index[7] += rows_per_tasklet;
                i++;
            }

            base_index[7] = cur_index[7];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_7 = DPU_MRAM_HEAP_POINTER;
                mram_offset_7 += read_buf[cur_index[7]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_7, &write_buf[7][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[7]++;
                cur_num_rows[7]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_7 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_7 += (answer_begin_index+base_index[7]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[7], mram_offset_7, read_len*cur_num_rows[7]);
                            
                    base_index[7] = cur_index[7];
                    cur_num_rows[7] = 0;
                }
            }      
        }
        break;

        case 8:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<8){
                cur_index[8] += rows_per_tasklet+1;
                i++;
            }

            while(i<8){
                cur_index[8] += rows_per_tasklet;
                i++;
            }

            base_index[8] = cur_index[8];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_8 = DPU_MRAM_HEAP_POINTER;
                mram_offset_8 += read_buf[cur_index[8]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_8, &write_buf[8][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[8]++;
                cur_num_rows[8]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_8 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_8 += (answer_begin_index+base_index[8]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[8], mram_offset_8, read_len*cur_num_rows[8]);
                            
                    base_index[8] = cur_index[8];
                    cur_num_rows[8] = 0;
                }
            }            
        }
        break;

        case 9:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<9){
                cur_index[9] += rows_per_tasklet+1;
                i++;
            }

            while(i<9){
                cur_index[9] += rows_per_tasklet;
                i++;
            }

            base_index[9] = cur_index[9];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

             while(j<till){
                    
                mram_offset_9 = DPU_MRAM_HEAP_POINTER;
                mram_offset_9 += read_buf[cur_index[9]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_9, &write_buf[9][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[9]++;
                cur_num_rows[9]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_9 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_9 += (answer_begin_index+base_index[9]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[9], mram_offset_9, read_len*cur_num_rows[9]);
                            
                    base_index[9] = cur_index[9];
                    cur_num_rows[9] = 0;
                }
            }       
        }
        break;

        case 10:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<10){
                cur_index[10] += rows_per_tasklet+1;
                i++;
            }

            while(i<10){
                cur_index[10] += rows_per_tasklet;
                i++;
            }

            base_index[10] = cur_index[10];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_10 = DPU_MRAM_HEAP_POINTER;
                mram_offset_10 += read_buf[cur_index[10]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_10, &write_buf[10][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[10]++;
                cur_num_rows[10]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_10 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_10 += (answer_begin_index+base_index[10]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[10], mram_offset_10, read_len*cur_num_rows[10]);
                            
                    base_index[10] = cur_index[10];
                    cur_num_rows[10] = 0;
                }
            }
        }
        break;

        case 11:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<11){
                cur_index[11] += rows_per_tasklet+1;
                i++;
            }

            while(i<11){
                cur_index[11] += rows_per_tasklet;
                i++;
            }

            base_index[11] = cur_index[11];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_11 = DPU_MRAM_HEAP_POINTER;
                mram_offset_11 += read_buf[cur_index[11]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_11, &write_buf[11][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[11]++;
                cur_num_rows[11]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_11 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_11 += (answer_begin_index+base_index[11]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[11], mram_offset_11, read_len*cur_num_rows[11]);
                            
                    base_index[11] = cur_index[11];
                    cur_num_rows[11] = 0;
                }
            }
        }
        break;

        case 12:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<12){
                cur_index[12] += rows_per_tasklet+1;
                i++;
            }

            while(i<12){
                cur_index[12] += rows_per_tasklet;
                i++;
            }

            base_index[12] = cur_index[12];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

            while(j<till){
                    
                mram_offset_12 = DPU_MRAM_HEAP_POINTER;
                mram_offset_12 += read_buf[cur_index[12]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_12, &write_buf[12][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[12]++;
                cur_num_rows[12]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_12 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_12 += (answer_begin_index+base_index[12]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[12], mram_offset_12, read_len*cur_num_rows[12]);
                            
                    base_index[12] = cur_index[12];
                    cur_num_rows[12] = 0;
                }
            }            
        }
        break;

        case 13:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<13){
                cur_index[13] += rows_per_tasklet+1;
                i++;
            }

            while(i<13){
                cur_index[13] += rows_per_tasklet;
                i++;
            }

            base_index[13] = cur_index[13];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
                
            int j=0; 

             while(j<till){
                    
                mram_offset_13 = DPU_MRAM_HEAP_POINTER;
                mram_offset_13 += read_buf[cur_index[13]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_13, &write_buf[13][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[13]++;
                cur_num_rows[13]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_13 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_13 += (answer_begin_index+base_index[13]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[13], mram_offset_13, read_len*cur_num_rows[13]);
                            
                    base_index[13] = cur_index[13];
                    cur_num_rows[13] = 0;
                }
            }       
        }
        break;

        case 14:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<14){
                cur_index[14] += rows_per_tasklet+1;
                i++;
            }

            while(i<14){
                cur_index[14] += rows_per_tasklet;
                i++;
            }

            base_index[14] = cur_index[14];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_14 = DPU_MRAM_HEAP_POINTER;
                mram_offset_14 += read_buf[cur_index[14]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_14, &write_buf[14][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[14]++;
                cur_num_rows[14]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_14 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_14 += (answer_begin_index+base_index[14]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[14], mram_offset_14, read_len*cur_num_rows[14]);
                            
                    base_index[14] = cur_index[14];
                    cur_num_rows[14] = 0;
                }
            }
        }
        break;

        case 15:{

            barrier_wait(&my_barrier);

            int i=0;
            int till;

            while(i<num_with_one_more && i<15){
                cur_index[15] += rows_per_tasklet+1;
                i++;
            }

            while(i<15){
                cur_index[15] += rows_per_tasklet;
                i++;
            }

            base_index[15] = cur_index[15];

            if(i<num_with_one_more)
                till = rows_per_tasklet+1;
            else
                till = rows_per_tasklet;
            
            int j=0; 

            while(j<till){
                    
                mram_offset_15 = DPU_MRAM_HEAP_POINTER;
                mram_offset_15 += read_buf[cur_index[15]]*nr_cols*sizeof(uint64_t);

                mram_read(mram_offset_15, &write_buf[15][(j%cols_per_buf)*nr_cols], read_len);

                cur_index[15]++;
                cur_num_rows[15]++;
                    
                j++; 

                if((j*nr_cols)%256==0 || j==till){
                    mram_offset_15 = DPU_MRAM_HEAP_POINTER;
                    mram_offset_15 += (answer_begin_index+base_index[15]*nr_cols)*sizeof(uint64_t);

                    mram_write(write_buf[15], mram_offset_15, read_len*cur_num_rows[15]);
                            
                    base_index[15] = cur_index[15];
                    cur_num_rows[15] = 0;
                }
            }
        }
        break;
    }

    return 0;
}
