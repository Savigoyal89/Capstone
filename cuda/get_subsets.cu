#include <iostream>
using namespace std;
#define THREADS_PER_BLOCK 16
#define MAX_ELEMENT 26

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)



__host__ __device__ int get_r_steps_product(int n, int steps) {
  int out = 1;
  for (int i = 0; i < steps; i++) {
    out = out * (n - i);
  }
  return out;
}

__host__ __device__ int nCr(int n, int r) {
  if (r == 0) {
    return 1;
  } else if (r == 1) {
    return n;
  } else {
    return (get_r_steps_product(n, r) / get_r_steps_product(r, r));
  }
}

__host__ int convert_subset_to_number(int *subset, int subset_size) {
  int sum = 0;
  for (int i = subset_size - 1; i >= 0; i--) {
    sum = sum * MAX_ELEMENT + subset[i];
  }
  return sum;
}

__host__ void convert_number_to_subset(int number, int *subset) {
  int index = 0;
  while (number > 0) {
    subset[index] = number % MAX_ELEMENT;
    number = number / MAX_ELEMENT;
    index++;
  }
}

__host__ void get_combinations(int *arr, int arr_index, int *current, int current_index,
                      int elements_size, int subset_size, int *output,
                      int &output_index) {
  // Current combination is ready, print it
  if (current_index == subset_size) {
    // Copy current to output at output_index
    output[output_index] = convert_subset_to_number(current, subset_size);
    output_index++;
    return;
  }
  // When no more elements are there to put in data[]
  if (arr_index >= elements_size) return;
  // Case 1: Exclude the arr_index
  get_combinations(arr, arr_index + 1, current, current_index, elements_size,
                   subset_size, output, output_index);

  // Case 2: Include the arr_index into the current_index
  current[current_index] = arr[arr_index];
  get_combinations(arr, arr_index + 1, current, current_index + 1,
                   elements_size, subset_size, output, output_index);
}


int main(int argc, char *argv[]) {
  // Declare host copies
  int arr[] = {0,1, 2, 3, 4, 5, 6};
  int array_size = 7;
  int subset_size = 4;
  int subset[subset_size] = {0};
  // Calculate the output size
  int output_size = nCr(array_size, subset_size);
  int output[output_size+1] = {0};
  output[0]= subset_size;
  int output_index = 1;
  get_combinations(arr, 0, subset, 0, array_size, subset_size, output,
                   output_index);
  printf("Output Index: %d \n",output_index);
  printf("First index of output : %d\n",output[0]);
  for (int i = 1; i < output_index; i++) {
    convert_number_to_subset(output[i],subset);
    for (int j = 0; j < subset_size; j++) {
      printf("%d ", subset[j]);
    }
    printf("\n");
  }
}

