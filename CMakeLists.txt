cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_eigen)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(test_eigen source/test_eigen.cpp)
target_link_libraries(
  test_eigen
  Eigen3::Eigen)
set_target_properties(test_eigen PROPERTIES CXX_STANDARD 11)
