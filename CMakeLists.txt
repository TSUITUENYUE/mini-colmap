cmake_minimum_required(VERSION 3.20)

project(hw4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(COLMAP REQUIRED)
find_package(Ceres REQUIRED)
find_package(VTK REQUIRED)



include_directories(${COLMAP_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${VTK_USE_DIRS} include src)
link_directories(${COLMAP_LINK_DIRS})

add_executable(mini_colmap mini_colmap.cpp
        src/read_data.cpp
        src/estimate.cpp
        src/optimize.cpp
        src/triangulation.cpp
        src/EPnP.cpp
        src/bundle_adjust.cpp)

target_link_libraries(mini_colmap
        ${COLMAP_LIBRARIES}
        ${CERES_LIBRARIES}
        ${VTK_LIBRARIES})

