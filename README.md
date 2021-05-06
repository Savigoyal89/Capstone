# Capstone Project

## Background
In mathematics, the Prouhet–Tarry–Escott(PTE) problem asks for two disjoint multisets A and B of n integers each, whose first k power sum symmetric polynomials are all equal. In other words 

for each integer i from 1 to a given k. An example solution for n = 6 is given by the two sets { 0, 5, 6, 16, 17, 22 } and { 1, 2, 10, 12, 20, 21 } :
01 + 51 + 61 + 161 + 171 + 221 = 11 + 21 + 101 + 121 + 201 + 211
02 + 52 + 62 + 162 + 172 + 222 = 12 + 22 + 102 + 122 + 202 + 212
03 + 53 + 63 + 163 + 173 + 223 = 13 + 23 + 103 + 123 + 203 + 213
Any pair A, B that satisfies the equation is called a solution of the PTE Problem; this is denoted by A =k B . If k = n − 1, then the solution is called ideal and n is called the size of this ideal solution. 
For this project, we restrict set A and set B as mentioned above to be integers from 0 to 25 and the value of n to be less than or equal to 25. 

## Goal
In this project, I plan to implement a solution based on CUDA that identifies the ideal solutions of the Prouhet-Tarry-Escott problem in CUDA and compare the performance with the baseline implementation using MPI library.

## Instructions to run MPI
```shell
foo@bar:~$ cd mpi
foo@bar:~$ mpicc -o get_subsets get_subsets.cpp       
foo@bar:~$ sbatch mpi.sbatch
```

## Instructions to run CUDA
```shell
foo@bar:~$ cd cuda
foo@bar:~$ nvcc get_subsets.cu -o get_subsets           
foo@bar:~$ srun get_subsets
```


