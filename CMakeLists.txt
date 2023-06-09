cmake_minimum_required(VERSION 2.8.12)
project(msckf_vio CXX)
# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_BUILD_TYPE Release)
# add_compile_options(-std=c++14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++17 -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fpic")

set(CMAKE_CXX_STANDARD 14) 


if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

LIST(APPEND CMAKE_MODULE_PATH /home/ldd/package/g2o/cmake_modules)
SET(G2O_ROOT /home/ldd/package/g2o/)
set(Sophus_DIR "/home/ldd/package/Sophus/build")
find_package(g2o REQUIRED)

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  random_numbers
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  std_srvs
  Sophus REQUIRED
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

##################
## ROS messages ##
##################
add_message_files(
  FILES

  FeatureMeasurement.msg
  CameraMeasurement.msg
  TrackingInfo.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES msckf_vio image_processor
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions random_numbers message_runtime
    image_transport cv_bridge message_filters pcl_conversions
    pcl_ros std_srvs
  DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

# Msckf Vio
add_library(msckf_vio
  src/msckf_vio.cpp
  src/utils.cpp
)
add_dependencies(msckf_vio
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(msckf_vio
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  g2o_core g2o_stuff
)

# Msckf Vio nodelet
add_library(msckf_vio_nodelet
  src/msckf_vio_nodelet.cpp
)
add_dependencies(msckf_vio_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(msckf_vio_nodelet
  msckf_vio
  ${catkin_LIBRARIES}
  g2o_core g2o_stuff
)

# Image processor
add_library(image_processor
  src/image_processor.cpp
  src/utils.cpp
)
add_dependencies(image_processor
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  g2o_core g2o_stuff
)

# Image processor nodelet
add_library(image_processor_nodelet
  src/image_processor_nodelet.cpp
)
add_dependencies(image_processor_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_nodelet
  image_processor
  ${catkin_LIBRARIES}
  g2o_core g2o_stuff
)

#############
## Install ##
#############

install(TARGETS
  msckf_vio msckf_vio_nodelet image_processor image_processor_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  # Feature initialization test
  catkin_add_gtest(test_feature_init
    test/feature_initialization_test.cpp
  )
  add_dependencies(test_feature_init
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(test_feature_init
    ${catkin_LIBRARIES}
  )

  # Math utils test
  catkin_add_gtest(test_math_utils
    test/math_utils_test.cpp
  )
endif()
