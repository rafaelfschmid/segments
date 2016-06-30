/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <chrono>
#include <iostream>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

void print(thrust::host_vector<int> h_vec) {
	std::cout << h_vec.size() << "\n";
	for (int i = 0; i < h_vec.size(); i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	thrust::host_vector<int> h_seg_aux(num_of_segments+1);
	for (i = 0; i < num_of_segments+1; i++)
		scanf("%d", &h_seg_aux[i]);

	scanf("%d", &num_of_elements);
	thrust::host_vector<int> h_vec(num_of_elements);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	thrust::host_vector<int> h_seg(num_of_elements);
	for (i = 0; i < num_of_segments; ++i){
		for(int j = h_seg_aux[i]; j < h_seg_aux[i+1]; ++j) {
			h_seg[j] = h_seg_aux[i];
		}
	}

	//print(h_seg); print(h_vec);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	thrust::device_vector<int> d_vec = h_vec;
	thrust::device_vector<int> d_seg = h_seg;

	cudaEventRecord(start);
	thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_seg.begin());
	thrust::sort_by_key(d_seg.begin(), d_seg.end(), d_vec.begin());
	cudaEventRecord(stop);

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	thrust::copy(d_seg.begin(), d_seg.end(), h_seg.begin());

	if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
	}
	else
		print(h_vec);

	return 0;
}
