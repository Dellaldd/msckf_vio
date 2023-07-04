# CMake generated Testfile for 
# Source directory: /home/ldd/msckf_real/src/msckf_vio
# Build directory: /home/ldd/msckf_real/src/msckf_vio/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_msckf_vio_gtest_test_feature_init "/home/ldd/msckf_real/src/msckf_vio/build/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/ldd/msckf_real/src/msckf_vio/build/test_results/msckf_vio/gtest-test_feature_init.xml" "--return-code" "/home/ldd/msckf_real/src/msckf_vio/build/devel/lib/msckf_vio/test_feature_init --gtest_output=xml:/home/ldd/msckf_real/src/msckf_vio/build/test_results/msckf_vio/gtest-test_feature_init.xml")
set_tests_properties(_ctest_msckf_vio_gtest_test_feature_init PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;98;catkin_run_tests_target;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;37;_catkin_add_google_test;/home/ldd/msckf_real/src/msckf_vio/CMakeLists.txt;162;catkin_add_gtest;/home/ldd/msckf_real/src/msckf_vio/CMakeLists.txt;0;")
add_test(_ctest_msckf_vio_gtest_test_math_utils "/home/ldd/msckf_real/src/msckf_vio/build/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/ldd/msckf_real/src/msckf_vio/build/test_results/msckf_vio/gtest-test_math_utils.xml" "--return-code" "/home/ldd/msckf_real/src/msckf_vio/build/devel/lib/msckf_vio/test_math_utils --gtest_output=xml:/home/ldd/msckf_real/src/msckf_vio/build/test_results/msckf_vio/gtest-test_math_utils.xml")
set_tests_properties(_ctest_msckf_vio_gtest_test_math_utils PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;98;catkin_run_tests_target;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;37;_catkin_add_google_test;/home/ldd/msckf_real/src/msckf_vio/CMakeLists.txt;174;catkin_add_gtest;/home/ldd/msckf_real/src/msckf_vio/CMakeLists.txt;0;")
subdirs("gtest")
