cmake_minimum_required(VERSION 3.16)
project(HandwrittenTextRecognition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(htr_preprocess
    cpp/src/main.cpp
    cpp/src/preprocessing.cpp
    cpp/src/segmentation.cpp
)

target_include_directories(htr_preprocess PRIVATE cpp/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(htr_preprocess PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_executable(pipeline_tests
        tests/pipeline_tests.cpp
        cpp/src/preprocessing.cpp
        cpp/src/segmentation.cpp
    )
    target_include_directories(pipeline_tests PRIVATE cpp/include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(pipeline_tests PRIVATE ${OpenCV_LIBS})
    add_test(NAME pipeline_tests COMMAND pipeline_tests)
endif()
