/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION

	__shared__ float sdata[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
   	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	if (i < size) {
        sdata[tid] = in[i] + in[i + blockDim.x];
   	 } else {
        sdata[tid] = 0;
    	}
   	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
	__syncthreads();
    }
	if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
     __shared__ float sdata[BLOCK_SIZE]; 

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < size) {
        sdata[tid] = in[i] + in[i + blockDim.x];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();  

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1){
        if (tid < s) {
 
           sdata[tid] += sdata[tid + s];
        }
 	__syncthreads();
     }
 if (tid < 32) {
    volatile float* vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}   

     if (tid == 0) {
        out[blockIdx.x] = sdata[0];
     }
}



