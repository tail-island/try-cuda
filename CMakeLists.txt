cmake_minimum_required(VERSION 3.18)

project(try-cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(TBB         REQUIRED)
find_package(Thrust      REQUIRED)

thrust_create_target(Thrust)

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
