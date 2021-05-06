
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cmath>
using std::cerr;
using std::endl;
#include <fstream>
using std::ofstream;
#include <cstdlib> // for exit function


using std::vector;

using namespace std;
// Driver code



void get_combinations(int *arr, int n, int r,
                      std::vector<int> &current, int i,
                      std::vector<std::vector<int>> &output) {
    // Current combination is ready, print it
    if (current.size() == r) {
        output.push_back(current);
        cout << endl;
        return;
    }
    // When no more elements are there to put in data[]
    if (i >= n)
        return;

    // current is included, put next at next location
    current.push_back(arr[i]);
    get_combinations(arr, n, r, current, i + 1, output);
    current.pop_back();

    // current is excluded, replace it with next (Note that
    // i+1 is passed, but index is not changed)
    get_combinations(arr, n, r, current, i + 1, output);
}

void get_subsets(int *input_elements, int num_input_elements, int subset_size,
                 std::vector<std::vector<int>> &output) {
    // A temporary vector to store
    // all combination one by onesubset_size
    if(num_input_elements >= subset_size){
                std::vector<int> current;
                get_combinations(input_elements, num_input_elements, subset_size, current,
                0, output);
        }
    return;
}

int fact(int st, int en)
{
    int res = 1;
    for (int i = en; i <=st ; i++)
        res = res * i;
    return res;
}


int nCr(int n, int r)
{
    return fact(n,n-r+1) / fact(r,1) ;
}

/**
 * convert subsets of a given size from an input set of elements to 1 d array.
 * @param vec Input all subsets 2 dimensional array
 * @return output dimensional array subset_one_d_array[], which contains all the subsets in a unique subset number form
 */
int *twoDVecCombToNumConversionArr(std::vector<std::vector<int>> vec) {
    int total_size =vec.size();
    int base =26;
    int *newarr=new int[total_size];
    int partial_size;
    for (int i=0; i<total_size; i++) {
        int sum =0;
        int exp = vec[i].size()-1;
        for (int j=0; j<vec[i].size();j++){
            int base_power = pow(base, exp);
            sum +=  base_power * vec[i][j];
            exp-=1;
        }
        newarr[i] = sum;
    }
    return newarr;
}


int* twoDVecToArr(std::vector<std::vector<int>> vec, int total_size) {

    int* newarr=new int[total_size];
    int index = 0;
    for (int i=0; i<vec.size(); i++) {
        for (int j=0; j<vec[i].size();j++){
            newarr[index] = vec[i][j];
            index +=1;
        }
//        newarr[index] = 100;
//        index+=1;
    }
    return newarr;
}

/**
 * convert unique subset number into subset vector of a given size.
 * @param num Input unique subset number
 * @param count_of_digits Input number of solution every process contain
 * @return subset vector formed from the number.
 */

std::vector<int> conversionNumberToSubset(int num, int count_of_digits){
    int base =26;
    std::vector<int> subset_comb(count_of_digits);
    for(int i =0; i<count_of_digits; i++){
        subset_comb[i] =0;
    }
    int index = count_of_digits-1;
    while(num>0){
        int rem =num%26;
        num = num/26;
        subset_comb[index] = rem;
        index-=1;
    }
    return subset_comb;
}

int *create_nums_arr(int num_elements) {
    int *arr_nums = (int *) malloc(sizeof(int) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        arr_nums[i] = i;
    }
    return arr_nums;
}


void get_set_diff(int *input_elements,  std::vector<int> subset,
                  int num_input_elements, int subset_size, int *out) {
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < num_input_elements) {
        if (j<subset_size && (input_elements[i] == subset[j])) {
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
}

/**
 * Compute the number of counts sent to each process.
 * @param num_of_process input total number of processes
 * @param count_of_subsets_for_the_process Input combination_count i.e nCr(num_elements,subset_size)
 * @return count_send i.e count array that contains the number of elements to send to each process.
 */
int *makeSendCountArrayToScatterElements( int num_of_process, int count_of_subsets_for_the_process ){

    int floor =0;
    floor =  count_of_subsets_for_the_process/num_of_process;
    int rem_subsets_left  = count_of_subsets_for_the_process%num_of_process;
    int *count_arr =new int[num_of_process];
    for(int i =0; i< num_of_process; i++){
        count_arr[i] = floor;
    }
    if (rem_subsets_left!=0){
        for(int i =0; i< rem_subsets_left; i++){
            count_arr[i]+=1;
        }
    }
    return count_arr;
}


/**
 * Compute the number of counts sent to each process.
 * @param num_of_process input total number of processes.
 * @param count_arr Input An array that contains the number of elements to send to each process.
 * @return displacements_send that contains displacement to apply to the message sent to each process.
 */

int *makeSendDisplacementArray(int num_of_process, int *count_arr){

    int *dis_arr =new int[num_of_process];
    int start_index=0;
    for(int i=0; i<num_of_process; i++){
        dis_arr[i] = start_index;
        start_index += count_arr[i];
    }
    return dis_arr;
}


int main(int argc, char *argv[]) {
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    double mpiWtime = MPI_Wtime();  /*get the time just before work to be timed*/
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_of_process = world_size;
    int num_elements = 26;
    int subset_size = 6;

    int *nums_arr = NULL;
    vector<vector<int>> primarySubsets;
    int *subset_one_d_array = NULL;
    nums_arr =create_nums_arr(num_elements);
    if(world_rank== 0){
        get_subsets(nums_arr, num_elements, subset_size,primarySubsets);
        subset_one_d_array = twoDVecCombToNumConversionArr(primarySubsets);
    }

    int combination_count = nCr(num_elements,subset_size);
    int *counts_send =NULL;
    int *displacements_send =NULL;
    counts_send = makeSendCountArrayToScatterElements(num_of_process, combination_count);
    displacements_send = makeSendDisplacementArray(num_of_process, counts_send);

    int *buffer_recv;
    int buffer_recv_length =0;

    buffer_recv_length = counts_send[world_rank];

    buffer_recv = (int*)malloc(sizeof(int) * buffer_recv_length);
    if(world_rank== 0) {
        MPI_Scatterv(subset_one_d_array, counts_send, displacements_send, MPI_INT, buffer_recv, buffer_recv_length, MPI_INT, 0,
                     MPI_COMM_WORLD);
    }
    else{
        MPI_Scatterv(NULL, NULL, NULL,MPI_INT, buffer_recv, buffer_recv_length, MPI_INT, 0, MPI_COMM_WORLD);
    }
    free(subset_one_d_array);
    free(counts_send);
    free(displacements_send);

    int *pte_array;
    int pte_array_size =0;
    std::vector<std::vector<int>> pte_comb;

    for (int i = 0; i < buffer_recv_length; i++) {
        std::vector<int> subset_comb = conversionNumberToSubset(
                buffer_recv[i], subset_size);

        int smaller_number_elemination_count = subset_comb[0]-1;
        int temp_size = num_elements - subset_size - smaller_number_elemination_count-1;
        int *temp_array = new int[temp_size];
        get_set_diff(nums_arr, subset_comb, num_elements, subset_size,temp_array);
        std::vector<std::vector<int>> partner_comb;
        get_subsets(temp_array, temp_size, subset_size, partner_comb);

        for (int j = 0; j < partner_comb.size(); j++) {
            bool pte = is_ideal_PTE(subset_comb, partner_comb[j], subset_size);
             if (pte == true) {
                std::vector<int> full_pte_vector;

                full_pte_vector.reserve(subset_size +
                                                subset_size); // preallocate memory
                full_pte_vector.insert(full_pte_vector.end(),
                                       subset_comb.begin(),
                                       subset_comb.end());
                full_pte_vector.insert(full_pte_vector.end(),
                                       partner_comb[j].begin(),
                                       partner_comb[j].end());

                pte_comb.push_back(full_pte_vector);
                pte_array_size+=(subset_size*2);

            }
        }
    }

    int pte_array_final_size = pte_array_size;
    pte_array = twoDVecToArr(pte_comb,pte_array_final_size);
    MPI_Barrier(MPI_COMM_WORLD);

    int *gatherv_receive_counts = new int[num_of_process];
    int nelements[1] = {pte_array_final_size};

// Each process tells the root how many elements it holds
    MPI_Gather(nelements, 1, MPI_INT, gatherv_receive_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

// Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[num_of_process];

// Displacement for the first chunk of data - 0
    for (int i = 0; i < num_of_process; i++)
        disps[i] = (i > 0) ? (disps[i-1] + gatherv_receive_counts[i-1]) : 0;

// Place to hold the gathered data
// Allocate at root only

    int *alldata = NULL;
    int all_data_size = 0;
    if (world_rank == 0){
        all_data_size = disps[num_of_process-1]+gatherv_receive_counts[num_of_process-1];
        alldata = new int[all_data_size];

    }
// Collect everything into the root
    MPI_Gatherv(pte_array, pte_array_final_size, MPI_INT, alldata, gatherv_receive_counts, disps, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        ofstream outdata; // outdata is like cin
        outdata.open("prouhetTarryEscott.dat"); // opens the file
        if( !outdata ) { // file couldn't be opened
            cerr << "Error: file could not be opened" << endl;
            exit(1);
        }

        outdata << "Ideal PTE solutions for n = 6 are: ";

        for (int is = 0; is < all_data_size; is++) {
            if(is%(subset_size*2)==0){
                outdata <<endl;
                outdata << alldata[is];
            }
            else if(is%subset_size==0) {
                outdata << " = ";
                outdata << alldata[is];
            }

            else{
                outdata <<", ";
                outdata << alldata[is];

            }

        }
        mpiWtime = MPI_Wtime() - mpiWtime;
        outdata <<endl;
        outdata <<endl;
        outdata <<endl;
        outdata << "Number of Processes = "<<world_size<<"\t"<< "Total time taken by processes = "  << mpiWtime <<"seconds";
        outdata.close();

    }
    free(alldata);
    free(buffer_recv);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return EXIT_SUCCESS;

    }