cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files
include_directories(/home/neraoof/onnxruntime/cmake/external/eigen)
include_directories(/home/neraoof/ort_build/onnxruntime-linux-x64-1.0.0/include)


find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/neraoof/ort_build/onnxruntime-linux-x64-1.0.0/lib)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
