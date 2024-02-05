cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(simulator LANGUAGES CUDA C CXX)

# deps
add_subdirectory(3rd)

include(FetchContent)

FetchContent_Declare(
  cudatest
  GIT_REPOSITORY https://github.com/shikui08/cudatest.git
  GIT_TAG master
)
FetchContent_Populate(cudatest)

# target
file(GLOB_RECURSE SRC_FILES src/*.cu src/*.cpp src/*.h)
add_executable(simulator ${SRC_FILES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
include_directories(simulator /usr/local/cuda-12.1/include/)
target_compile_options(simulator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -std=c++20
                       --expt-relaxed-constexpr
                       >)

set_target_properties(simulator PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 89) 
target_link_libraries(simulator 3rd_libs)
target_link_libraries(simulator GLU)

# install
install (TARGETS simulator DESTINATION bin)
install (TARGETS 3rd_libs
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib
RUNTIME DESTINATION bin)