cmake_minimum_required(VERSION 4.3.4)
project(Talos CXX)

set(CMAKE_CXX_STANDARD 23)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

execute_process(COMMAND brew --prefix libomp
                OUTPUT_VARIABLE LIBOMP
                OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(Hephaestus ops.cpp)
target_compile_options(Hephaestus     PUBLIC -Xpreprocessor -fopenmp)
target_include_directories(Hephaestus PUBLIC ${LIBOMP}/include)
target_link_libraries(Hephaestus      PUBLIC ${LIBOMP}/lib/libomp.dylib)

add_executable(ElderScroll main.cpp)
target_link_libraries(ElderScroll PRIVATE Hephaestus)

add_custom_target(run COMMAND ElderScroll DEPENDS ElderScroll
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
