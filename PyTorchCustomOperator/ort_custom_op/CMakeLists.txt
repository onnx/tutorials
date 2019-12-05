cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(<PATH_TO_EIGEN_DIR>)
include_directories(<PATH_TO_ONNXRUNTIME_INCLUDE_DIR>)

#Include path to header files for Custom Op Test
include_directories(<PATH_TO_ONNXRUNTIME_TEST_UTIL_INCLUDE_DIR>)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS <PATH_TO_ONNXRUNTIME_LIB>)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})