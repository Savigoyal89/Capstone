cmake_minimum_required(VERSION 3.17)
project(Capstone)

set(CMAKE_CXX_STANDARD 14)

add_executable(Capstone main.cpp baseline/library.cpp baseline/library.h)