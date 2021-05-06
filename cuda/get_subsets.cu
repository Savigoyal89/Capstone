#include <iostream>
#include <cmath>
using namespace std;
#define THREADS_PER_BLOCK 512
#define MAX_ELEMENT 26
#define NUM_ELEMENTS_SUBSET 6
#define OFFSET 6

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

__host__ __device__ int get_combinations(int *arr, int arr_index, int *current, int current_index,
                      int elements_size, int subset_size, int *output,
                      int output_index) {
  // Current combination is ready, print it
  if (current_index == subset_size) {
    // Copy current to output at output_index
    output[output_index] = convert_subset_to_number(current, subset_size);
    return output_index + 1;
  }
  // When no more elements are there to put in data[]
  if (arr_index >= elements_size) return output_index;
  // Case 1: Exclude the arr_index
  output_index = get_combinations(arr, arr_index + 1, current, current_index, elements_size,
                   subset_size, output, output_index);

  // Case 2: Include the arr_index into the current_index
  current[current_index] = arr[arr_index];
  output_index = get_combinations(arr, arr_index + 1, current, current_index + 1,
                   elements_size, subset_size, output, output_index);
  return output_index;
}



__device__ int get_sum_of_power(int* set, int set_size, int power, int setNum, int* sum_of_square) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  sum_of_square[index]=0;
    for (int i = 0; i < set_size; i++) {
	    int power_val = 1;
            for (int j =0; j < power; j++){
	       power_val = power_val*set[i];
	    }
	    int new_sum = sum_of_square[index] + power_val;
	      if (setNum==60465){
	        printf("SetNum: %d , Set number: %d , %d + %d = %d\n",setNum,set[i],sum_of_square[index],power_val,new_sum);
	    }
	   sum_of_square[index] = new_sum;   
    }
    return sum_of_square[index];
}

__device__ bool is_ideal_PTE(int setNum1, int setNum2,int* sum_of_square) {
    if (setNum1 <0 || setNum2 <0){
       return false;
    }
    bool print_results= false;
    if((setNum1 == 64080 && setNum2 == 60465) || (setNum2 == 64080 && setNum1 == 60465)){
    	print_results = true;
    }
    int set1[NUM_ELEMENTS_SUBSET] = {0};
    int set2[NUM_ELEMENTS_SUBSET] = {0};
    convert_number_to_subset(setNum1,set1);
    convert_number_to_subset(setNum2,set2);
   if (print_results){
       printf("IS ideal PTE called for %d and %d\n",setNum1,setNum2);
   }
    for (int i = 1; i < NUM_ELEMENTS_SUBSET; i++) {
	      int sumPowerSet1 = get_sum_of_power(set1, NUM_ELEMENTS_SUBSET, i,setNum1, sum_of_square);
	      int sumPowerSet2 = get_sum_of_power(set2, NUM_ELEMENTS_SUBSET, i,setNum2, sum_of_square);
        if (sumPowerSet1 != sumPowerSet2){
	    if (print_results){
	    	printf("Result returned false for power %d with sum1= %d and sum2 = %d\n",i,sumPowerSet1, sumPowerSet2);
	     }
      return false;
       }
    }
    if (print_results){
      printf("Result returned true\n");
    }
    return true;
}

__global__ void get_ideal_pte_combinations(int* input, int* output,int* max_index, int* sum_of_square){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Get idea PTE solution for thread id:%d having max_index:%d\n",index,*max_index);
  if (index >= *max_index){
  	return;
  }
  int input_subset_number = input[index];
  int offset=0;
  for (int i=index+1; i < *max_index;i++){
    bool is_solution = is_ideal_PTE(input_subset_number, input[i], sum_of_square);
    if (is_solution){
        printf("Found ideal PTE match: %d -- %d which will be updated for output index:%d and %d\n",input_subset_number,input[i],index*OFFSET + offset,index*OFFSET + offset+1);
        output[index*OFFSET + offset] = input_subset_number;
        output[index*OFFSET + offset + 1] = input[i];
        offset +=2;
      }
    }
  printf("Check completed for subset number %d\n",input_subset_number);
  //delete[] partner_subsets;
}

__host__ void print_subsets(int* subsets, int num_subsets, int subset_size ){
  int subset[subset_size] = {0};
  for (int i = 0; i < num_subsets; i++) {
    convert_number_to_subset(subsets[i], subset);
    printf("Subset number: %d --- ",subsets[i]);
    for (int j = 0; j < subset_size; j++) {
      printf("%d ", subset[j]);
    }
    printf("\n");
  }
}

__host__ void print_output(int* subsets, int num_subsets ){
  int subset[NUM_ELEMENTS_SUBSET] = {0};
  for (int i = 0; i < num_subsets-1; i=i+2) {
    if (subsets[i]==0 && subsets[i+1]==0){
    	continue;
    }
    printf("Start index: %d ",i );
    convert_number_to_subset(subsets[i], subset);
    for (int j = 0; j < NUM_ELEMENTS_SUBSET; j++) {
      printf("%d ", subset[j]);
    }
    printf(" -- ");
    convert_number_to_subset(subsets[i+1], subset);
    for (int j = 0; j < NUM_ELEMENTS_SUBSET; j++) {
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
  size_t limit = 256;

  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);
  printf("cudalimitprintffifosize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitDevRuntimeSyncDepth);
  printf("cudaLimitDevRuntimeSyncDepth: %u\n", (unsigned)limit);

  int *d_subsets, *d_output, *d_max_index, *d_sum_of_squares;  // device copies
  int size = num_subsets * sizeof(int);

  // Allocate space for device copies
  cudaMalloc((void**) &d_subsets, size);
  cudaMalloc((void**) &d_sum_of_squares, size);
  cudaMalloc((void**) &d_max_index, sizeof(int));
  cudaMalloc((void**) &d_output, size*OFFSET);
  cudaMemcpy(d_subsets, h_subsets,size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_index, &num_subsets,sizeof(int), cudaMemcpyHostToDevice);

  // Run the functions on device
  int num_blocks = num_subsets/THREADS_PER_BLOCK +1;
  printf("Number of blocks =%d with threads per block=%d for total number of subsets=%d\n", num_blocks,THREADS_PER_BLOCK,num_subsets);
  get_ideal_pte_combinations<<<num_blocks,THREADS_PER_BLOCK>>>(d_subsets,d_output,d_max_index,d_sum_of_squares);
  cudaDeviceSynchronize();

  // Copy from device to host
  int h_output[num_subsets*OFFSET] = {0};
  cudaMemcpy(h_output, d_output, size*OFFSET, cudaMemcpyDeviceToHost);
  printf("Printing ideal PTE output for %d element subset \n", NUM_ELEMENTS_SUBSET);
  print_output(h_output,num_subsets*OFFSET);

  // Clean up
  cudaFree(d_subsets);
  cudaFree(d_output);
  cudaFree(d_max_index);
  return 0;
}

