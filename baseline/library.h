//
// Created by Savi Goyal on 2/17/21.
//
#include <vector>
#ifndef CAPSTONE_LIBRARY_H
#define CAPSTONE_LIBRARY_H

/**
 * Create input element set of a given size(num_elements) containing elements
 * from 0 to num_elements-1.
 * @param num_elements
 * @return
 */
int *create_nums_arr(int num_elements);

/**
 * Gets all the subsets of given size from an input set of elements.
 * @param input_elements Input elements
 * @param num_input_elements Number of input elements
 * @param subset_size Number of elements in a subset
 * @param output Vector of subsets
 */
void get_subsets(int *input_elements, int num_input_elements, int subset_size,
                 std::vector<std::vector<int>> &output);

///**
// * Get the difference between the two sets. In other words, get the elements present in
// * the input_elements but not in the subset.
// * @param input_elements  Input element set
// * @param subset The subset of the elements to excluded in the output.
// * @param num_input_elements Number of input elements
// * @param subset_size Number of elements in a subset
// * @param out Output set of elements.
// *
// *
// */
//void get_set_diff(const int *input_elements,  std::vector<int> subset,
//                  int num_input_elements, int subset_size, int *out);

/**
 *  Check if two sets of elements are ideal PTE solutions.
 * @param set1 Input set 1
 * @param set2 Input set 2
 * @param set_size Set size
 * @return true if the sets are ideal PTE, else false.
 */
bool is_ideal_PTE(std::vector<int> set1, std::vector<int> set2, int set_size);

#endif //CAPSTONE_LIBRARY_H
