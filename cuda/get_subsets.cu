#include <iostream>
using namespace std;
#define THREADS_PER_BLOCK 1024
#define MAX_ELEMENT 7
#define NUM_ELEMENTS_SUBSET 3

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


__host__ __device__ void get_nums_array(int* arr){
    for (int i =0; i < MAX_ELEMENT;i++){
      arr[i]=i;
    }
}

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

__host__ __device__ int convert_subset_to_number(int *subset, int subset_size) {
  int sum = 0;
  for (int i = subset_size - 1; i >= 0; i--) {
    sum = sum * MAX_ELEMENT + subset[i];
  }
  return sum;
}

__host__ __device__ void convert_number_to_subset(int number, int *subset) {
  int index = 0;
  while (number > 0) {
    subset[index] = number % MAX_ELEMENT;
    number = number / MAX_ELEMENT;
    index++;
  }
}

__host__ __device__ void get_combinations(int *arr, int arr_index, int *current, int current_index,
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

__device__ int get_set_diff(int *input_elements, int* subset,int *out) {
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < MAX_ELEMENT) {
        if (j<NUM_ELEMENTS_SUBSET && (input_elements[i] == subset[j])) {
            j++;
        }
        else {
            if (input_elements[i]>subset[0]) {
                out[k] = input_elements[i];
                k++;
            }
        }
        i++;
    }
  return k;
}

__global__ void get_ideal_pte_combinations(int* input, int* output){
  int index = threadIdx.x ;
  int input_subset_number = input[index];
  int subset[NUM_ELEMENTS_SUBSET]= {0};
  convert_number_to_subset(input_subset_number, subset);
  int set_diff[MAX_ELEMENT -  NUM_ELEMENTS_SUBSET] = {0};
  int arr[MAX_ELEMENT] = {0};
  get_nums_array(arr);
  int set_diff_size = get_set_diff(arr,subset,set_diff);
  output[index]  = input_subset_number;
}

__host__ void print_subsets(int* subsets, int num_subsets, int subset_size ){
  int subset[subset_size] = {0};
  for (int i = 0; i < num_subsets; i++) {
    convert_number_to_subset(subsets[i], subset);
    for (int j = 0; j < subset_size; j++) {
      printf("%d ", subset[j]);
    }
    printf("\n");
  }
}

__host__ void get_input_subsets(int subset_size, int* subsets){
  int subset[subset_size] = {0};
  // Calculate the output size
  int output_index = 0;
  int arr[MAX_ELEMENT] = {0};
  get_nums_array(arr);
  printf("Subset size: %d\n", subset_size);
  get_combinations(arr, 0, subset, 0, MAX_ELEMENT, subset_size, subsets,output_index);
}

int main(int argc, char *argv[]) {
  // Declare host copies
  int num_subsets = nCr(MAX_ELEMENT, NUM_ELEMENTS_SUBSET);
  int h_subsets[num_subsets] = {0};
  get_input_subsets(NUM_ELEMENTS_SUBSET, h_subsets);
  printf("Printing output from CPU with size: %d\n", num_subsets);
  print_subsets(h_subsets,num_subsets,NUM_ELEMENTS_SUBSET);
  // Initiate device copies
  int *d_subsets, *d_output;  // device copies
  int size = num_subsets * sizeof(int);

  // Allocate space for device copies
  cudaMalloc((void**) &d_subsets, size);
  cudaMalloc((void**) &d_output, size);

  // Copy from host to device
  cudaMemcpy(d_subsets, h_subsets,size, cudaMemcpyHostToDevice);

  // Run the functions on device
  get_ideal_pte_combinations<<<1,num_subsets>>>(d_subsets,d_output);

  // Copy from device to host
  h_subsets[num_subsets] = {0};
  cudaMemcpy(h_subsets, d_output, size, cudaMemcpyDeviceToHost);
  printf("Printing output from GPU with size: %d\n",num_subsets);
  print_subsets(h_subsets, num_subsets,NUM_ELEMENTS_SUBSET);

  // Clean up
  cudaFree(d_subsets);
  cudaFree(d_output);
  return 0;

}

