cmake_minimum_required(VERSION 4.3.4)
project(Talos)
add_executable(ElderScroll main.cpp)

#Add the libraries 
add_library(Hephaestus ops.cpp)
target_link_libraries(ElderScroll PRIVATE Hephaestus)

target_compile_options(ElderScroll PRIVATE -Wall -Wextra -fsanitize=address -g)
target_link_options(ElderScroll PRIVATE -fsanitize=address)

