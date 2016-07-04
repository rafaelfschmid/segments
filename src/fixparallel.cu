/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

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
#include <iostream>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

__global__ void pre_fix_sum(uint *d_vec, uint *d_seg, uint num_segments) {//,int* d_value) {

	uint begin = d_seg[blockIdx.x];
	uint end = d_seg[blockIdx.x + 1];
	uint size = end - begin;
	//printf("begin=%d\n", begin);
	//extern __shared__ int max_value[];

	__shared__ uint max_value[2 * BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	//unsigned int start = 2 * blockIdx.x * blockDim.x;
	max_value[t] = d_vec[begin + t];
	max_value[blockDim.x + t] = d_vec[begin + blockDim.x + t];
	syncthreads();

	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		syncthreads();
		if (t < stride && max_value[t] < max_value[t + stride] )
			max_value[t] = max_value[t + stride];
	}

	d_vec[t] = max_value[t];
	d_vec[begin + blockDim.x + t] = max_value[blockDim.x + t];

	/*int id = blockIdx.x * blockDim + threadIdx.x;
	//if(id < end) {
	for (; id < end; id += BLOCK_SIZE) {
		if (d_vec[id] < d_vec[id])
			max_value[i] = h_vec[i];
	}

	mostSignificantBit = (uint) log2((double) maxValue) + 1;

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = d_seg[i]; j < d_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] += segIndex;
		}
	}*/
	//}

}

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec_aux = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec_aux[i]);
		h_value[i] = i;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_value, *d_value_out, *d_vec, *d_vec_out, *d_seg;
	void *d_temp = NULL;
	size_t temp_bytes = 0;

	cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint mostSignificantBit = 0;
	for (uint k = 0; k < EXECUTIONS; k++) {

		for (i = 0; i < num_of_elements; i++) {
			h_vec[i] = h_vec_aux[i];
			h_value[i] = i;
		}

		std::chrono::high_resolution_clock::time_point start1 =
				std::chrono::high_resolution_clock::now();
		uint maxValue = 0;
		for (i = 0; i < num_of_elements; i++) {
			if (maxValue < h_vec[i])
				maxValue = h_vec[i];
		}

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));

		pre_fix_sum<<<1, BLOCK_SIZE>>>(d_vec, d_seg, num_of_segments);

		cudaTest(cudaMemcpy(h_vec, d_vec, mem_size_vec,	cudaMemcpyDeviceToHost));
		printf("max1=%d, max2=%d", h_vec[0], h_vec[2*BLOCK_SIZE]);
/*
		mostSignificantBit = (uint) log2((double) maxValue) + 1;

		for (i = 0; i < num_of_segments; i++) {
			for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
				uint segIndex = i << mostSignificantBit;
				h_vec[j] += segIndex;
			}
		}
		std::chrono::high_resolution_clock::time_point stop1 =
				std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<
				std::chrono::duration<double>>(stop1 - start1);

		cudaTest(
				cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		cudaTest(
				cudaMemcpy(d_value, h_value, mem_size_vec,
						cudaMemcpyHostToDevice));

		if (temp_bytes == 0) {
			cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec,
					d_vec_out, d_value, d_value_out, num_of_elements);
			cudaMalloc((void **) &d_temp, temp_bytes);
		}
		cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out,
				d_value, d_value_out, num_of_elements);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		cudaTest(
				cudaMemcpy(h_vec, d_vec_out, mem_size_vec,
						cudaMemcpyDeviceToHost));

		start1 = std::chrono::high_resolution_clock::now();
		for (i = 0; i < num_of_segments; i++) {
			for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
				uint segIndex = i << mostSignificantBit;
				h_vec[j] -= segIndex;
			}
		}
		stop1 = std::chrono::high_resolution_clock::now();
		time_span += std::chrono::duration_cast < std::chrono::duration<double>
				> (stop1 - start1);

		if (ELAPSED_TIME == 1) {
			std::cout << time_span.count() * 1000 << "\n";
		}
*/

		cudaDeviceSynchronize();
	}
	cudaFree(d_seg);
	cudaFree(d_vec);
	cudaFree(d_vec_out);
	cudaFree(d_value);
	cudaFree(d_value_out);
	cudaFree(d_temp);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg);
	free(h_vec);
	free(h_vec_aux);
	free(h_value);

	return 0;
}
