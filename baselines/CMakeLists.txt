cmake_minimum_required(VERSION 3.1)
project(MyCGALProject)

find_package(CGAL REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main CGAL::CGAL)