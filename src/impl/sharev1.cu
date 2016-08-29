/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <bitset>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

void print(int* h_data, int n) {
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << h_data[i] << " ";
	}
	std::cout << "\n";
}

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}



/*
 * 256 threads per block
 * 4 elements per thread
 * = 1024 elements per block
 * = n/1024 blocks
 */

template<int BITS_NUMBER = 64>
__global__ void radix_sort(int *d_vec, int *d_seg, int num_segments) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int begin = d_seg[bx];
	int end = d_seg[bx + 1];
	int size = end - begin;

	__shared__ int s_vec[BLOCK_SIZE];
	__shared__ int s_aux[BLOCK_SIZE];
	__shared__ int s_pref_sum_one[BLOCK_SIZE];
	__shared__ int s_pref_sum_zero[BLOCK_SIZE];

	for (int k = 0; k < size; k += BLOCK_SIZE) {
		int threadIndexGlobal = begin + k + tx;

		int block = BLOCK_SIZE;
		if(BLOCK_SIZE+k > size)
			block = size-k;

		if (threadIndexGlobal < end) {
			s_vec[tx] = d_vec[threadIndexGlobal];
			__syncthreads();

			int i, j;
			int exp = 0;

			for (j = 0; j < BITS_NUMBER; j++) {

				int x = (s_vec[tx] >> exp) & 1;
				s_pref_sum_one[tx] = x;
				s_pref_sum_zero[tx] = 1-x;
				__syncthreads();

				for (i = 1; i < block; i*=2) {
					int index = tx + i;
					if (index < block) {
						int one = s_pref_sum_one[tx] + s_pref_sum_one[index];
						int zero = s_pref_sum_zero[tx] + s_pref_sum_zero[index];
						__syncthreads();
						s_pref_sum_one[index] = one;
						s_pref_sum_zero[index] = zero;
						__syncthreads();
					}
				}

				x = (s_vec[tx] >> exp) & 1;
				int index = (x) * (s_pref_sum_one[tx] + s_pref_sum_zero[block-1] - 1)
						+ (1 - x) * (s_pref_sum_zero[tx] - 1);
				s_aux[index] = s_vec[tx];
				__syncthreads();

				s_vec[tx] = s_aux[tx];
				__syncthreads();

				exp++;
			}
			d_vec[threadIndexGlobal] = s_aux[tx];
		}
	}
}

int main(int argc, char **argv) {

	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	int mem_size_seg = sizeof(int) * (num_of_segments + 1);
	int *h_seg = (int *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(int) * num_of_elements;
	int *h_vec = (int *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

//	print(h_seg, num_of_segments + 1);	print(h_vec, num_of_elements);

// Allocate device memory
	int *d_seg, *d_vec;

	cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
//cudaTest(cudaMalloc((void **) &d_aux, mem_size_vec));

// copy host memory to device
	cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

	// Setup execution parameters
//	int devID = 0;
//	cudaDeviceProp deviceProp;
//	cudaTest(cudaGetDeviceProperties(&deviceProp, devID));
//	unsigned int multiprocessor_number = deviceProp.multiProcessorCount;
//	//unsigned int grid_blocks_max_x = deviceProp.maxGridSize[0];
//	//unsigned int sharedMemoryTotal = deviceProp.sharedMemPerBlock/(sizeof(int));
//
	int blocksize = BLOCK_SIZE; //num_of_elements / num_of_segments;
	//if (blocksize > 1024)
	//	blocksize = 1024;

	dim3 threads(blocksize, 1);
	//dim3 grid(num_of_segments / blocksize + 1, 1);
	dim3 grid(num_of_segments, 1);

	std::chrono::high_resolution_clock::time_point start =
			std::chrono::high_resolution_clock::now();

	radix_sort<EXP_BITS_SIZE> <<<grid, threads>>>(d_vec, d_seg,
			num_of_segments);
	cudaDeviceSynchronize();
	std::chrono::high_resolution_clock::time_point stop =
			std::chrono::high_resolution_clock::now();
	cudaTest(cudaPeekAtLastError());
	std::chrono::duration<double> time_span = std::chrono::duration_cast<
			std::chrono::duration<double>>(stop - start);

	cudaTest(cudaMemcpy(h_seg, d_seg, mem_size_seg, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost));

	//print(h_seg, num_of_segments + 1);
	//print(h_vec, num_of_elements);
	//print(h_seg, 10);
	//print(h_vec, 1000);
	if (ELAPSED_TIME == 1)
//		std::cout << "It took me " << time_span.count() * 1000
//				<< " miliseconds.\n";
		;
	else
		print(h_vec, num_of_elements);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	free(h_seg);
	free(h_vec);
	cudaFree(d_seg);
	cudaFree(d_vec);

	return 0;
}

/**
 * 	// cudaDeviceReset causes the driver to clean up all state. While
 // not mandatory in normal operation, it is good practice.  It is also
 // needed to ensure correct operation when the application is being
 // profiled. Calling cudaDeviceReset causes all profile data to be
 // flushed before the application exits
 cudaDeviceReset();
 */

/*
 printf("thread=%d | aux=%d %d %d %d\n", bx * blockDim.x + tx, d_aux[begin], d_aux[begin + 1], d_aux[begin + 2],	d_aux[begin + 3]);
 int devID = 0;
 cudaDeviceProp deviceProp;
 cudaTest(cudaGetDeviceProperties(&deviceProp, devID));
 unsigned int multiprocessorNumber = deviceProp.multiProcessorCount;
 unsigned int sharedMemoryTotal = deviceProp.sharedMemPerBlock/(sizeof(int));
 std::cout << "multiprocessorNumber: " << multiprocessorNumber << "\n";
 std::cout << "sharedMemoryTotal: " << sharedMemoryTotal << "\n";
 std::cout << "numberOfSegmentsPerBlock: " << sharedMemoryTotal << "\n";
 */
