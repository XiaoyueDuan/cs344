//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <device_launch_parameters.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

 // Histogram
__global__ void hist_cal(unsigned int *d_out, const unsigned int *const d_in, const size_t numBins, const unsigned int pos, const size_t size)
{
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= size)
	{
		printf("Overflow!!!\n");
		return;
	}	

	// bin = (lum[i] - lumMin) / lumRange * numBins
	unsigned int mask = (numBins - 1) << pos;
	int bin = (d_in[myId] & mask)>>pos;
	atomicAdd(&(d_out[bin]), 1);
}

// Scan
__global__ void scan_cal(unsigned int *d_cdf, const unsigned int *d_bins, const size_t numBins)
{	
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= numBins)
		return;

	int tid = threadIdx.x;
	extern __shared__ unsigned int adata[];
	adata[tid] = d_bins[myId];
	__syncthreads();

	unsigned int interval=0;
	int current_cdf= adata[tid];
	for (int level = 0; interval < numBins; ++level)
	{
		interval = 1 << level;
		if (myId >= interval)		
			current_cdf += adata[tid - interval];

		__syncthreads();
		adata[tid] = current_cdf;
		__syncthreads();
	}

	d_cdf[myId]= adata[tid];
	__syncthreads();
} 

// Compact
__global__ void map_cal(bool *d_out_bool, unsigned int *d_out_value, const unsigned int *const d_in, 
						const unsigned int mask,
						const unsigned int num, 
						const size_t size)
{
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= size)
		return;	
	
	d_out_bool[myId] = ((d_in[myId]&mask)==num);
	d_out_value[myId] = (d_out_bool[myId]) ? 1 : 0;
}

__global__ void map_address(unsigned int *d_out, const unsigned int *d_cdf, const bool *d_bool, const size_t size)
{
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= size)
		return;

	if(d_bool[myId])
		d_out[myId] = d_cdf[myId];

	__syncthreads();
}

void compact(unsigned int *d_out, const unsigned int *const d_in, const size_t numBins, const size_t pos, const size_t size)
{
	// pos: which digits will be test
	//									x = 1 0 1 1 1 1 0 0
	//		pos = 0, numBins = 2(2^1),					  ¡ü(last)
	//		pos = 2, numBins = 2(2^1),				  ¡ü
	//		pos = 0, numBins = 8(2^3),				  ¡ü¡ü¡ü
	
	unsigned int threads = 1024;
	unsigned int blocks = (size + threads - 1) / threads;

	bool *d_bool; 
	checkCudaErrors(cudaMalloc((void **)&d_bool, sizeof(bool)*size));
	unsigned int *d_value;
	checkCudaErrors(cudaMalloc((void **)&d_value, sizeof(unsigned int)*size));
	unsigned int *d_pos;
	checkCudaErrors(cudaMalloc((void **)&d_pos, sizeof(unsigned int)*size));

	unsigned int mask;
	for (int num=0; num<numBins; ++num)
	{
		mask = (numBins - 1)<<pos;
		cudaMemset((void **)&d_bool, false, sizeof(bool)*size);
		cudaMemset((void **)&d_value, 0, sizeof(unsigned int)*size);

		// 1) Compute predicate & 2) Scan_in array
		map_cal<<<threads, blocks>>>(d_bool, d_value, d_in, mask, num << pos, size);// pause here to see whether num changes

		// 3) Exclusive-sum_scan
		scan_cal << <threads, blocks >> >(d_pos, d_value, size);

		// 4) Scatter input into outter using address
		map_address << <threads, blocks >> >(d_out, d_value, d_bool, size);
	}

	checkCudaErrors(cudaFree(d_bool));
	checkCudaErrors(cudaFree(d_value));
	checkCudaErrors(cudaFree(d_pos));
}

__global__ void specific_address(unsigned int *d_out_value, const unsigned int *const d_in_value, 
								unsigned int *d_out_pos, const unsigned int *const d_in_pos,
								const unsigned int *d_cdfs, const unsigned int *d_relative_pos,
								const size_t numBins, const unsigned int pos, const size_t size)
{
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= size)
		return;

	unsigned int mask = (numBins - 1) << pos;
	int bin = (d_in_value[myId] & mask) >> pos;
	unsigned int address = d_cdfs[myId] + d_relative_pos[myId];
	d_out_value[address] = d_in_value[myId];
	d_out_pos[address] = d_in_pos[myId];

	//__syncthreads();
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
	unsigned int numBins = 1;
	numBins = 1<< numBins;
	
	unsigned int threads = 1024;
	unsigned int blocks = (numElems + threads - 1) / threads;

	unsigned int *d_hist;
	checkCudaErrors(cudaMalloc((void **)&d_hist,sizeof(unsigned int)*numBins));
	unsigned int *d_cdfs;
	checkCudaErrors(cudaMalloc((void **)&d_cdfs, sizeof(unsigned int)*numBins));

	unsigned int *d_relative_pos;
	checkCudaErrors(cudaMalloc((void **)&d_relative_pos, sizeof(unsigned int)*numElems));
	unsigned int *d_tmp_value;
	checkCudaErrors(cudaMalloc((void **)&d_tmp_value, sizeof(unsigned int)*numElems));
	unsigned int *d_tmp_pos;
	checkCudaErrors(cudaMalloc((void **)&d_tmp_pos, sizeof(unsigned int)*numElems));

	unsigned int mask;
	for (unsigned int i = 0; i < sizeof(unsigned int) * 8; i += numBins)
	{
		checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins));
		checkCudaErrors(cudaMemset(d_cdfs, 0, sizeof(unsigned int)*numBins));

		//	1) Histogram of the number of occurrences of each digit
		hist_cal <<< threads, blocks >>> (d_hist, d_inputVals, numBins, i, numElems);

		//	2) Exclusive Prefix Sum of Histogram
		scan_cal <<< threads, blocks >>> (d_cdfs, d_hist, numBins);

		//	3) Determine relative offset of each digit
		//		For example[0 0 1 1 0 0 1]
		//				 ->[0 1 0 1 2 3 2]
		compact(d_relative_pos, d_inputVals, numBins, i, numElems);

		//	4) Combine the results of steps 2 & 3 to determine the final
		//	   output location for each element and move it there
		specific_address(d_tmp_value, d_inputVals, 
			d_tmp_pos, d_inputPos,
			d_cdfs, d_relative_pos,
			numBins, i, numElems);

		checkCudaErrors(cudaMemcpy(d_inputVals, d_tmp_value, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_inputPos, d_tmp_pos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		// without move positions!!!!!!!!!!!!!!!!!!!!!!!
	}
	// input value being modified??????????????????????????????????????
	// double const meanings???????????????????????????????????????????
	checkCudaErrors(cudaMemcpy(d_outputVals, d_tmp_value, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_tmp_pos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaFree(d_hist));
	checkCudaErrors(cudaFree(d_cdfs));
	checkCudaErrors(cudaFree(d_relative_pos));
	checkCudaErrors(cudaFree(d_tmp_value));
}
