# Set the minimum required CMake version and project details
cmake_minimum_required(VERSION 3.18)
project(GPU_Linear_Systems LANGUAGES CXX CUDA)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Check for CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "CUDA architectures not specified. CMake will attempt to detect them automatically.")
    message(WARNING "To specify architectures manually, use -DCMAKE_CUDA_ARCHITECTURES=<arch>")
endif()

# Find required packages
find_package(cudss 0.5 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Define MKL_ROOT cache variable without a default path
set(MKL_ROOT "" CACHE PATH "Path to MKL root directory")

# Check if MKL_ROOT is set and exists
if(NOT MKL_ROOT OR NOT EXISTS "${MKL_ROOT}/include/mkl.h")
    message(FATAL_ERROR "MKL_ROOT not set or MKL headers not found at specified path: '${MKL_ROOT}'. Please set the MKL_ROOT variable using -DMKL_ROOT=<path_to_mkl>")
endif()
message(STATUS "Using MKL root: ${MKL_ROOT}")

# Set MKL include and lib directories based on MKL_ROOT
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

# Find Intel OpenMP Runtime Library (libiomp5)
find_library(IOMP5_LIB NAMES iomp5
             HINTS ENV LD_LIBRARY_PATH "${MKL_ROOT}/lib/intel64" "${MKL_ROOT}/lib" /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
             PATH_SUFFIXES lib lib64 lib/intel64)
if(NOT IOMP5_LIB)
    message(FATAL_ERROR "Could not find Intel OpenMP Runtime Library (libiomp5.so). Please ensure Intel oneAPI is installed and MKL_ROOT is set correctly, or specify the library path manually using -DIOMP5_LIB=<path>.")
endif()
message(STATUS "Found Intel OpenMP library: ${IOMP5_LIB}")

# Define MKL libraries once
set(MKL_LIBRARIES
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    ${IOMP5_LIB}
)

# Include the main project directory to make utils.h available to all targets
include_directories(${CMAKE_SOURCE_DIR})

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Add executables
# --------------------
# Task Pardiso
add_executable(task_pardiso task_pardiso.cpp)
# Configure CPU-based target for Pardiso
target_include_directories(task_pardiso PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(task_pardiso PRIVATE ${MKL_LIBRARIES})

# --------------------
# Task CuDSS
set_source_files_properties(task_cudss.cpp PROPERTIES LANGUAGE CUDA)

add_executable(task_cudss task_cudss.cpp)
# Configure CUDA DSS target
target_include_directories(task_cudss PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(task_cudss PRIVATE cudss)
# Add compile definitions for CUDA version to cuDSS target
target_compile_definitions(task_cudss PRIVATE
    CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
    CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
)


# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")