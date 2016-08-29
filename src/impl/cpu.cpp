/*

 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include <time.h>
#include <chrono>

void print(std::vector<uint> h_vec) {
	std::cout << "\n";
	for (uint i = 0; i < h_vec.size(); i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	std::vector<uint> h_seg(num_of_segments + 1);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	std::vector<uint> h_vec(num_of_elements);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	std::chrono::high_resolution_clock::time_point start =
				std::chrono::high_resolution_clock::now();
	for (i = 0; i < num_of_segments; ++i) {
		std::stable_sort(h_vec.begin() + h_seg[i], h_vec.begin() + h_seg[i + 1]);
	}
	std::chrono::high_resolution_clock::time_point stop =
				std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<
				std::chrono::duration<double>>(stop - start);

	if (ELAPSED_TIME == 1) {
			std::cout << time_span.count()*1000 << "\n";
		}
		else
			print(h_vec);

	return 0;
}
