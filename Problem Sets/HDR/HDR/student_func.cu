/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.h"

// Reduce
__global__ void reduce_calMin(float* d_out, const float* const d_in)
{	
	// 1. Use shared Mem
	extern __shared__ float sdata[];
	
	int myId = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	
	sdata[tid]=d_in[myId];
	__syncthreads();
	
	// 2. Reduce
	int length = (int)blockDim.x;	
	for (int i = 0; length>1; ++i)
	{
		if(tid<length/2)
			sdata[tid] = min(sdata[tid], sdata[length-1-tid]);
		__syncthreads();

		length = (length + 1) / 2;
	}
	if(tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce_calMax(float* d_out, const float* const d_in)
{	
	// 1. Use shared Mem
	extern __shared__ float sdata[];
	
	int myId = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	
	sdata[tid]=d_in[myId];
	__syncthreads();
	
	// 2. Reduce
	int length = (int)blockDim.x;
	for (int i = 0; length>1; ++i)
	{								
		if (tid<(length / 2))
			sdata[tid] = max(sdata[tid], sdata[length - 1 - tid]);
		__syncthreads();

		length = (length + 1) / 2;
	}
	
	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

void reduce(const float* const d_in,
			const size_t size,
			float &min_logLum, float &max_logLum)
{		
	float * d_intermediate, *d_out;

	// 1. Max reduce
	int threads = 1024;
	int blocks = (size + threads - 1) / threads;		
	
	checkCudaErrors(cudaMalloc((void **) &d_intermediate, sizeof(float) * blocks)); 
	checkCudaErrors(cudaMemset((void **) d_intermediate, 0.0f, sizeof(float) * blocks));
	checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(float)));	
	
	reduce_calMax << <blocks, threads, threads * sizeof(float) >> >(d_intermediate, d_in);
	
	threads=blocks;
	blocks=1;
	reduce_calMax << <blocks, threads, threads * sizeof(float) >> >(d_out, d_intermediate);
	checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	
	// 2. Min reduce	
	threads = 1024;
	blocks = (size + threads - 1) / threads;			
	
	checkCudaErrors(cudaMemset((void **) d_intermediate, 1.0f, sizeof(float) * blocks));	
	reduce_calMin << <blocks, threads, threads * sizeof(float) >> >(d_intermediate, d_in);
	
	threads = blocks;
	blocks = 1;
	reduce_calMin << <blocks, threads, threads * sizeof(float) >> >(d_out, d_intermediate);
	checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	// Clean up
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_intermediate));
}	

// Histogram
__global__ void hist_cal(unsigned int *d_out, const float* const d_in, float min_logLum, float range_logLum, const size_t numBins, const size_t size)
{
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= size)
	{
		printf("Overflow!!!\n");
		return;
	}	

	// bin = (lum[i] - lumMin) / lumRange * numBins
	int bin = (d_in[myId] - min_logLum) / range_logLum * numBins;
	bin = bin <= (numBins - 1) ? bin : (numBins - 1);
	atomicAdd(&(d_out[bin]), 1);
}

// Scan
__global__ void scan_cal(unsigned int *d_cdf, unsigned int *d_bins, const size_t numBins)
{	
	int myId = blockDim.x*blockIdx.x + threadIdx.x;
	if (myId >= numBins)
		return;

	int tid = threadIdx.x;
	extern __shared__ unsigned int adata[];
	adata[tid] = d_bins[myId];
	__syncthreads();

	int interval=0;
	int current_cdf= adata[tid];
	for (int level = 0; interval < numBins; ++level)
	{
		interval = powf(2, level);
		if (myId >= interval)		
			current_cdf += adata[tid - interval];

		__syncthreads();
		adata[tid] = current_cdf;
		__syncthreads();
	}

	d_cdf[myId]= adata[tid];
	__syncthreads();
}

// Overall
void your_histogram_and_prefixsum(const float* const d_logLuminance,
								  unsigned int* const d_cdf,
								  float &min_logLum,
								  float &max_logLum,
								  const size_t numRows,
								  const size_t numCols,
								  const size_t numBins)
{ 
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // Step 1: Find the minimum and maximum value
  int ARRAY_BYTES = numRows * numCols;
  // declare GPU memory pointers  
  reduce(d_logLuminance, ARRAY_BYTES, min_logLum, max_logLum);

  // Step 2: Subtract minimum and maximum value to find the range
  float range_logLum = max_logLum - min_logLum;

  // Step 3: Generate a histogram of all the values
  unsigned int *d_bins;
  checkCudaErrors(cudaMalloc((void **)& d_bins, numBins * sizeof(unsigned int)));  
  checkCudaErrors(cudaMemset((void **)d_bins, 0, sizeof(unsigned int)*numBins));

  int threads = 1024;
  int blocks = (ARRAY_BYTES + threads - 1) / threads; 
    
  hist_cal << <blocks, threads >> > (d_bins, d_logLuminance, min_logLum, range_logLum, numBins, ARRAY_BYTES);
  
  // Step 4: Perform an exclusive scan (prefix sum) on the histogram to get
  //		 the cumulative distribution of luminance values
  threads = (numBins < 1024) ? numBins : 1024;
  blocks = (numBins + threads - 1) / threads;
  scan_cal << <blocks, threads, numBins* sizeof(unsigned int)>> >(d_cdf, d_bins, numBins);

  int h_cdf[1024];
  cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaFree(d_bins));
}
