cmake_minimum_required(VERSION 4.1)
project("CS:4700 Final Project" C CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT /opt/cuda)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)

add_executable(project
        src/pgm.c
        src/pgm.h
        src/kernels.c
        src/kernels.h
)
