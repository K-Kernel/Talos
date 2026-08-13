cmake_minimum_required(VERSION 4.3.4)
project(Talos)
add_executable(ElderScroll main.cpp)

#Add the libraries 
add_library(Hephaestus ops.cpp)
target_link_libraries(ElderScroll PRIVATE Hephaestus)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_SAN "Build with AddressSanitizer" OFF)
if(ENABLE_ASAN)
	target_compile_options(ElderScroll PRIVATE -Wall -Wextra -fsanitize=address -g)
	target_link_options(ElderScroll PRIVATE -fsanitize=address)
endif()

# Add make run
add_custom_target(run COMMAND ElderScroll DEPENDS ElderScroll WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

