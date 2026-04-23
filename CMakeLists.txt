cmake_minimum_required(VERSION 4.1)
project("CS:4700 Final Project" C CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT /opt/cuda)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)

add_executable(proc_cpu
        src/proc_cpu.c
        src/util.c
        src/util.h
)
target_link_libraries(proc_cpu m)

add_executable(proc_2d
        src/proc_2d.cu
        src/util.c
        src/util.h
)
target_link_libraries(proc_2d m)

add_executable(proc_seq
        src/proc_seq.cu
        src/util.c
        src/util.h
)
target_link_libraries(proc_seq m)
