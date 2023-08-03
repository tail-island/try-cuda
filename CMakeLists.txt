cmake_minimum_required(VERSION 3.18)

project(try-cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Thrust      REQUIRED)

thrust_create_target(Thrust)

add_executable(${PROJECT_NAME}
    main.cu
)

target_compile_features(${PROJECT_NAME} PRIVATE
    cxx_std_20
)

target_link_libraries(${PROJECT_NAME}
    CUDA::cudart
    Thrust
)
