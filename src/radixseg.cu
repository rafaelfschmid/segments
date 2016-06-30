/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================

 COMPILAR USANDO O SEGUINTE COMANDO:

 nvcc segmented_sort.cu -o segmented_sort -std=c++11 --expt-extended-lambda -I"/home/schmid/Dropbox/Unicamp/workspace/sorting_segments/moderngpu-master/src"

 */

#include <cub/util_allocator.cuh>
//#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

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
//#include <cstdlib>
#include <iostream>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXECUTIONS
#define EXECUTIONS 11
#endif


using namespace std::chrono;
using namespace cub;

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

void printSeg(int* host_data, uint num_seg, uint num_ele) {
	std::cout << "\n";
	for (uint i = 0; i < num_seg; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << num_ele << " ";
	std::cout << "\n";
}

int main(int argc, char** argv) {

	int num_of_segments;
	int num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	int *h_seg = (int *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_value, *d_value_out, *d_vec, *d_vec_out;
	int *d_seg;
	void *d_temp = NULL;
	size_t temp_bytes = 0;

	cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	for (uint i = 0; i < EXECUTIONS; i++) {

		// copy host memory to device
		cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

		if(temp_bytes == 0) {
			cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_vec,
					d_vec_out, d_value, d_value_out, num_of_elements,
					num_of_segments, d_seg, d_seg + 1);
			cudaTest(cudaMalloc((void **) &d_temp, temp_bytes));
		}
		cudaEventRecord(start);
		cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_vec,
				d_vec_out, d_value, d_value_out, num_of_elements,
				num_of_segments, d_seg, d_seg + 1);
		cudaEventRecord(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	cudaTest(cudaMemcpy(h_value, d_value_out, mem_size_seg, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost));

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
	free(h_value);

	return 0;
}

/***
 * SEGMENTED SORT FUNCIONANDO
 *
 *
 uint n = atoi(argv[1]);
 uint m = atoi(argv[2]);
 uint num_segments = n / m;
 mgpu::standard_context_t context;
 rand_key<int> func(m);

 mgpu::mem_t<int> segs = mgpu::fill_function(func, num_segments, context);
 //mgpu::mem_t<int> segs = mgpu::fill_random(0, n - 1, num_segments, true, context);
 std::vector<int> segs_host = mgpu::from_mem(segs);
 mgpu::mem_t<int> data = mgpu::fill_random(0, pow(2, NUMBER_BITS_SIZE), n,
 false, context);
 mgpu::mem_t<int> values(n, context);
 std::vector<int> data_host = mgpu::from_mem(data);

 //	print(segs_host); print(data_host);

 mgpu::segmented_sort(data.data(), values.data(), n, segs.data(),
 num_segments, mgpu::less_t<int>(), context);

 std::vector<int> sorted = from_mem(data);
 std::vector<int> indices_host = from_mem(values);

 std::cout << "\n";
 //print(segs_host);
 //	print(data_host); print(indices_host);
 *
 */
