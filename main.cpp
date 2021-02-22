#include <iostream>
#include <stdlib.h>
#include "baseline/library.h"

using namespace std;
// Driver code
int main() {
    int *nums_arr = NULL;
    int *nums_arr_subset = NULL;
    int num_elements = 10;
    int r = 4;
    nums_arr = create_nums_arr(num_elements);
    for (int i = 0; i < num_elements; i++) {
        cout << nums_arr[i] << " ";
    }
    cout << endl;

    nums_arr_subset = create_nums_arr(r);
    for (int i = 0; i < r; i++) {
        cout << nums_arr_subset[i] << " ";
    }
    cout << endl;
    int *output = (int *) malloc(sizeof(int) * (num_elements - r));
    get_set_diff(nums_arr, nums_arr_subset, num_elements, r, output);
    for (int i = 0; i < (num_elements - r); i++) {
        cout << output[i] << " ";
    }
    cout << endl;
    return 0;
}
