cmake_minimum_required(VERSION 3.18)

project(try-cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(TBB         REQUIRED)
find_package(Thrust      REQUIRED)

thrust_create_target(Thrust)

# reduction

add_executable(reduction
    reduction.cu
    utility.h
)

target_compile_features(reduction PRIVATE
    cxx_std_20
)

target_link_libraries(reduction
    CUDA::cudart
    TBB::tbb
    Thrust
)

# pi-cpu

add_executable(pi-cpu
    pi-cpu.cxx
    utility.h
)

target_compile_features(pi-cpu PRIVATE
    cxx_std_23
)

# pi-cuda

add_executable(pi-cuda
    pi-cuda.cu
    utility.h
)

target_compile_features(pi-cuda PRIVATE
    cxx_std_20
)

target_compile_options(pi-cuda PRIVATE
    --extended-lambda
)

target_link_libraries(pi-cuda
    CUDA::cudart
    CUDA::curand
    TBB::tbb
    Thrust
)
