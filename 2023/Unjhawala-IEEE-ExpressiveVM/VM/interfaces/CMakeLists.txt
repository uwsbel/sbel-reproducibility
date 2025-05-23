# This is a CMake example for Python
cmake_minimum_required(VERSION 3.8)

project(test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

#Set the cmake flags as O3 always
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE(PythonLibs)


INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
# Get the previous directory which has most of the header files
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../")
# INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../tires/")

SET(CMAKE_SWIG_FLAGS "")

SET_SOURCE_FILES_PROPERTIES(rom.i PROPERTIES CPLUSPLUS ON)
# SET_SOURCE_FILES_PROPERTIES(rom.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_LIBRARY(rom LANGUAGE python SOURCES ../../utils.cpp ../Eightdof.cpp rom.i)
SWIG_LINK_LIBRARIES(rom ${PYTHON_LIBRARIES})
