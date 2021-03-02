#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include "library.h"

using namespace std;

/**
 * Create input element set of a given size(num_elements) containing elements
 * from 0 to num_elements-1.
 * @param num_elements
 * @return
 */
int *create_nums_arr(int num_elements) {
    int *arr_nums = (int *) malloc(sizeof(int) * num_elements);
    assert(arr_nums != NULL);
    for (int i = 0; i < num_elements; i++) {
        arr_nums[i] = i+1;
    }
    return arr_nums;
}

/* arr[] ---> Input Array
n ---> Size of input array
r ---> Size of a combination to be printed
current---> Temporary vector to store current combination
i ---> index of current element in arr[]
output ---->vectors of vectors containing the whole subset
 */
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

/**
 * Gets all the subsets of a given size from an input set of elements.
 * @param input_elements Input elements
 * @param num_input_elements Number of input elements
 * @param subset_size Number of elements in a subset
 * @param output Vector of subsets
 */
void get_subsets(int *input_elements, int num_input_elements, int subset_size,
                 std::vector<std::vector<int>> &output) {
    // A temporary vector to store
    // all combination one by one
    std::vector<int> current;
    get_combinations(input_elements, num_input_elements, subset_size, current,
                     0, output);
    return;
}

/**
 * Get the difference between the two sets. In other words, get the elements present in
 * the input_elements but not in the subset.
 * @param input_elements  Input element set
 * @param subset The subset of the elements to excluded in the output.
 * @param num_input_elements Number of input elements
 * @param subset_size Number of elements in a subset
 * @param out Output set of elements.
 */
void get_set_diff(const int *input_elements,  std::vector<int> subset,
                  int num_input_elements, int subset_size, int *out) {

    int i = 0;
    int j = 0;
    int k = 0;
    while (i < num_input_elements) {
        if (j<subset_size && input_elements[i] == subset[j]) {
            j++;
        } else {
            out[k] = input_elements[i];
            k++;
        }
        i++;
    }
}

double get_sum_of_power(std::vector<int> set, int set_size, int power) {
    double sum = 0;
    for (int i = 0; i < set_size; i++) {
        sum += pow(set[i], power);
    }
    return sum;
}

/**
 *  Check if two sets of elements are ideal PTE solutions.
 * @param set1 Input set 1
 * @param set2 Input set 2
 * @param set_size Set size
 * @return true if the sets are ideal PTE, else false.
 */
bool is_ideal_PTE(std::vector<int> set1, std::vector<int> set2, int set_size) {
    for (int i = 1; i < set_size; i++) {
        if (get_sum_of_power(set1, set_size, i) !=
            get_sum_of_power(set2, set_size, i)) {
            return false;
        }
    }
    return true;
}