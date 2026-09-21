/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */



////////////////////////////////////////////////////////////////////////////////
// Common definition
////////////////////////////////////////////////////////////////////////////////
//Total number of possible data values
#define BIN_COUNT (1024) // Changed from 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))
//Machine warp size
#ifndef __DEVICE_EMULATION__
//G80's warp size is 32 threads
#define WARP_LOG_SIZE 5
#else
//Emulation currently doesn't execute threads in coherent groups of 32 threads,
//which effectively means warp size of 1 thread for emulation modes
#define WARP_LOG_SIZE 0
#endif
//Warps in thread block
#define  WARP_N 3
//Per-block number of elements in histograms
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)
#define IMUL(a, b) mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////

inline void addData1024(volatile __local uint *s_WarpHist, uint data, uint tag){
    uint count;
    do{
        count = s_WarpHist[data]  & 0x07FFFFFFU;
        count = tag | (count + 1);
        s_WarpHist[data] = count;
    }while(s_WarpHist[data] != count);
}

 __kernel void histogram1024Kernel(
                  __global uint *d_Result,
                  __global float *d_Data,
                     float minimum,
                     float maximum,
                    uint dataCount
                  ){
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    //Per-warp substorage storage
     int mulBase = (get_local_id(0) >> WARP_LOG_SIZE);
     const int warpBase = IMUL(mulBase, BIN_COUNT);
    __local unsigned int s_Hist[BLOCK_MEMORY];
     int test = 0;
     
//     if(get_global_id(0) == 0) {
//     for(int i = 0; i < 1024; i++) {
//             d_Result[i] = 0;
//         }
//     }
     const uint tag =  get_local_id(0) << (32 - WARP_LOG_SIZE);
    //Clear shared memory storage for current threadblock before processing
     for(uint i = get_local_id(0); i < BLOCK_MEMORY; i+=get_local_size(0)){
        s_Hist[i] = 0;
 }

    
    //Read through the entire input buffer, build per-warp histograms
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for(int pos = get_global_id(0); pos < dataCount; pos += get_global_size(0)){
        uint data4 = ((d_Data[pos] - minimum)/(maximum - minimum)) * BIN_COUNT;
        addData1024(s_Hist + warpBase, data4 & 0x3FFU, tag);
    }
    
    //Per-block histogram reduction
     // Sum is adding to index 0, pls fix
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for(int pos = get_local_id(0); pos < BIN_COUNT; pos += get_local_size(0)){
        uint sum = 0;
        for(int i = 0; i < BLOCK_MEMORY; i+= BIN_COUNT){
            sum += s_Hist[pos + i] & 0x07FFFFFFU;
        }
        atomic_add(d_Result+pos,sum);
    }
     
     
     
}



#define DIVISIONS               (1 << 10)
#define LOG_DIVISIONS	(10)
#define BUCKET_WARP_LOG_SIZE	(5)
#define BUCKET_WARP_N			(1)
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				(128)


int addOffset(volatile __local uint *s_offset, uint data, uint threadTag){
    uint count;

    do{
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    }while(s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__kernel void
bucketcount( global float *input, global int *indice, global uint *d_prefixoffsets, const int size, global float *l_pivotpoints)
{
    
	volatile __local uint s_offset[BUCKET_BLOCK_MEMORY];
    
    const uint threadTag = get_local_id(0) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0))
		s_offset[i] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
	for (int tid = get_global_id(0); tid < size; tid += numThreads) {
		float elem = input[tid];
        
		int idx  = DIVISIONS/2 - 1;
		int jump = DIVISIONS/4;
		float piv = l_pivotpoints[idx]; //s_pivotpoints[idx];
        
		while(jump >= 1){
			idx = (elem < piv) ? (idx - jump) : (idx + jump);
			piv = l_pivotpoints[idx]; //s_pivotpoints[idx];
			jump /= 2;
		}
		idx = (elem < piv) ? idx : (idx + 1);
        
		indice[tid] = (addOffset(s_offset + warpBase, idx, threadTag) << LOG_DIVISIONS) + idx;  //atomicInc(&offsets[idx], size + 1);
	}
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0))
		d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
}

__kernel void bucketprefixoffset(global uint *d_prefixoffsets, global uint *d_offsets, const int blocks) {
	int tid = get_global_id(0);
	int size = blocks * BUCKET_BLOCK_MEMORY;
	int sum = 0;
    
	for (int i = tid; i < size; i += DIVISIONS) {
		int x = d_prefixoffsets[i];
		d_prefixoffsets[i] = sum;
		sum += x;
	}
    
	d_offsets[tid] = sum;
}

__kernel void
bucketsort(global float *input, global int *indice, __global float *output, const int size, global uint *d_prefixoffsets,
		   global uint *l_offsets)
{
	volatile __local unsigned int s_offset[BUCKET_BLOCK_MEMORY];
    
	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);
    
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0)){
		s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (int tid = get_global_id(0); tid < size; tid += numThreads) {
       
		float elem = input[tid];
		int id = indice[tid];
		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
//        if(test == 2) {
//            printf("EDLLAWD %f", elem);
//        }
	}
}
