cmake_minimum_required(VERSION 3.10.2)
project(Capstone)

set(CMAKE_CXX_STANDARD 14)

add_executable(Capstone baseline/main.cpp baseline/library.cpp baseline/library.h)