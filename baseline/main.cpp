#include <iostream>
#include <stdlib.h>
#include "library.h"
#include <pthread.h>
#include <vector>

using namespace std;
// Driver code
#define NUM_THREADS 495

struct arg_struct {
    int thread_id;
    int num_elements;
    int *nums_array_copy =NULL;
    std::vector<int> primarySubset;
    std::vector<std::vector<int>> finalSubsets;

};

struct nested_args_struct {
    int thread_id;

    std::vector<int> primarySubset;
    std::vector<int> partnerSubset;
    std::vector<int> finalSubset;
};

void *get_final_sets(void *arguments){
    struct nested_args_struct *nested_args = (struct nested_args_struct *)arguments;


    if (is_ideal_PTE(nested_args->primarySubset, nested_args->partnerSubset, nested_args->partnerSubset.size())){

        for (int k = 0; k < nested_args->primarySubset.size(); k++) {
            nested_args ->finalSubset.push_back(nested_args->primarySubset[k]);
        }
        for (int k = 0; k < nested_args->partnerSubset.size(); k++) {
            nested_args ->finalSubset.push_back(nested_args->partnerSubset[k]);
        }

    }

   pthread_exit(NULL);
}

void get_partner_set(std::vector<std::vector<int>> partnerSubsets, arg_struct *args){

    int size =partnerSubsets.size();
    pthread_t nest_threads[size];
    struct nested_args_struct nested_args[size];

    for (int j = 0; j < size; j++) {

        nested_args[j].thread_id = j;
        nested_args[j].primarySubset = args ->primarySubset;
        nested_args[j].partnerSubset = partnerSubsets[j];
        if (pthread_create(&nest_threads[j], NULL, &get_final_sets, &nested_args[j]) != 0) {
            printf("Uh-oh!\n");
        }
    }
    for (int i = 0; i < size; i++) {
        pthread_join(nest_threads[i], NULL); /* Wait until thread is finished */
    }


    for (int i = 0; i < size; i++){
        if (nested_args[i].finalSubset.size() > 0) {
                args->finalSubsets.push_back(nested_args[i].finalSubset);

        }
    }
}

void *get_set_diff_helper(void *arguments){
    struct arg_struct *args = (struct arg_struct *)arguments;

    int new_array_size = args -> num_elements - args ->primarySubset.size();
    int *output = (int *) malloc(sizeof(int) * (new_array_size));
    get_set_diff(args ->nums_array_copy, args ->primarySubset, args ->num_elements, args ->primarySubset.size(), output);
    std::vector<std::vector<int>> partnerSubsets;
    get_subsets(output, new_array_size ,args ->primarySubset.size(), partnerSubsets);
    printf("PrimarySet of thread id: %d\n", args->thread_id);
    for (int j = 0; j < args ->primarySubset.size(); j++) {
        printf("%d ",args ->primarySubset[j]);
    }
    printf("\n");
    get_partner_set(partnerSubsets, args);

    //pthread_exit(NULL);
}


int main() {
    int *nums_arr = NULL;
    int num_elements = 12;

    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;

    nums_arr = create_nums_arr(num_elements);
    cout <<"Print elements in the main thread.\n";

    for (int i = 0; i < num_elements; i++) {
        cout << nums_arr[i] << " ";
    }
    std::vector<std::vector<int>> output;
    std::vector<std::vector<int>> finalOutput;

    //create all subsets max size 12 one one side
    std::vector<std::vector<int>> primarySubsets;
    for (int i = 4; i <5; i++) {

        get_subsets(nums_arr, num_elements, i,primarySubsets);
    }
    struct arg_struct args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].primarySubset = primarySubsets[i];
        args[i].num_elements = num_elements;
        args[i].nums_array_copy = create_nums_arr(num_elements);


        if (pthread_create(&threads[i], NULL, &get_set_diff_helper, &args[i]) != 0) {
            printf("Uh-oh!\n");
            return -1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL); /* Wait until thread is finished */
    }

    int k=0;
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 0; j < args[i].finalSubsets.size(); j++) {
            finalOutput.push_back(args[i].finalSubsets[j]);
        }
        printf("\n");
    }


    for (int i = 0; i <  finalOutput.size(); i++) {
        for (int j = 0; j <  finalOutput[i].size(); j++) {
            printf("%d ", finalOutput[i][j]);

        }
        k++;
        printf("\n");

    }
    cout << "No Of Solution found" <<k;
    cout << endl;
    pthread_exit(NULL);
}
