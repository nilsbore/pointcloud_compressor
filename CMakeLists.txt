set(CMAKE_CXX_FLAGS "-O4 -g -fPIC -std=c++0x")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(PCL_DIR "/opt/ros/groovy/share/pcl-1.6")
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#include_directories(/usr/include/eigen3)

add_library(pointcloud_compressor src/pointcloud_compressor.cpp)
add_library(pointcloud_compressor2 src/pointcloud_compressor2.cpp)
add_library(ksvd_decomposition src/ksvd_decomposition.cpp)
add_executable(test src/test.cpp)

target_link_libraries(pointcloud_compressor ${PCL_LIBRARIES})
target_link_libraries(pointcloud_compressor2 ${PCL_LIBRARIES})
target_link_libraries(test pointcloud_compressor pointcloud_viewer ${PCL_LIBRARIES})
