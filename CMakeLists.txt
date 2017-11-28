cmake_minimum_required(VERSION 3.8)
project(Trajectory_Prediction_GNB)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 11)

find_package(Boost)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

set(SOURCE_FILES main.cpp classifier.cpp classifier.h)
add_executable(Trajectory_Prediction_GNB ${SOURCE_FILES})