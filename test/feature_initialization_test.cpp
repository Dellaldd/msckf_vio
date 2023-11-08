/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <gtest/gtest.h>
#include <random_numbers/random_numbers.h>

#include <msckf_vio/cam_state.h>
#include <msckf_vio/feature.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <msckf_vio/image_processor.h>


using namespace std;
using namespace Eigen;
using namespace msckf_vio;

// Static member variables in CAMState class
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class
Feature::OptimizationConfig Feature::optimization_config;

vector<cv::Point2f> distortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs) {

  const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                      0.0, intrinsics[1], intrinsics[3],
                      0.0, 0.0, 1.0);

  vector<cv::Point2f> pts_out;
  if (distortion_model == "radtan") {
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
  } else {
    ROS_WARN_ONCE("The model %s is unrecognized, using radtan instead...",
                  distortion_model.c_str());
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  }

  return pts_out;
}

TEST(FeatureInitializeTest, sphereDistribution) {
  // Set the real feature at the origin of the world frame.
  Vector3d feature(0.5, 0.0, 0.0);
  int num = 2;
  vector<Isometry3d> cam_poses(6);//T_c_w

  cam_poses[0].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[0].translation() << 0.0,  0.0,  1.0;
  // Positive y axis.
  cam_poses[1].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[1].translation() << 0.0,  0.0,  2.0;
  // Negative x axis.
  cam_poses[2].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[2].translation() << 0.0,  0.0,  3.0;
  // Negative y axis.
  cam_poses[3].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[3].translation() << 0.0,  0.0,  -1.0;
  // Positive z axis.
  cam_poses[4].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[4].translation() << 0.0,  0.0,  -2.0;
  // Negative z axis.
  cam_poses[5].linear() << 1.0,  0.0, 0.0,
    0.0,  1.0,  0.0, 0.0, 0.0,  1.0;
  cam_poses[5].translation() << 0.0,  0.0,  -3.0;

  // load intrinsics
  Eigen::Matrix3d intrinsics0, intrinsics1;
  cv::Vec4d cam0_intrinsics(4);
  cam0_intrinsics[0] = 458.654;
  cam0_intrinsics[1] = 457.296;
  cam0_intrinsics[2] = 367.215;
  cam0_intrinsics[3] = 248.375;

  intrinsics0 << cam0_intrinsics[0], 0.0, cam0_intrinsics[2], 
    0.0, cam0_intrinsics[1], cam0_intrinsics[3], 
    0.0, 0.0, 1.0; 

  cv::Vec4d cam1_intrinsics(4);

  cam1_intrinsics[0] = 457.587;
  cam1_intrinsics[1] = 456.134;
  cam1_intrinsics[2] = 379.999;
  cam1_intrinsics[3] = 255.238;

  intrinsics1 << cam1_intrinsics[0], 0.0, cam1_intrinsics[2], 
          0.0, cam1_intrinsics[1], cam1_intrinsics[3], 
          0.0, 0.0, 1.0;

  cv::Vec4d cam0_distortion_coeffs(4);

  cam0_distortion_coeffs[0] = -0.28340811;
  cam0_distortion_coeffs[1] = 0.07395907;
  cam0_distortion_coeffs[2] = 0.00019359;
  cam0_distortion_coeffs[3] = 1.76187114e-05;

  cv::Vec4d cam1_distortion_coeffs(4);
  cam1_distortion_coeffs[0] = -0.28368365;
  cam1_distortion_coeffs[1] = 0.07451284;
  cam1_distortion_coeffs[2] = -0.00010473;
  cam1_distortion_coeffs[3] = -3.55590700e-05;

  string cam0_distortion_model = "radtan";
  string cam1_distortion_model = "radtan";

  // Set the camera states
  CamStateServer cam_states;
  for (int i = 0; i < num; ++i) {
    CAMState new_cam_state;
    new_cam_state.id = i;
    new_cam_state.time = static_cast<double>(i);
    new_cam_state.orientation = rotationToQuaternion(
        Matrix3d(cam_poses[i].linear().transpose())); // R_c_w
    new_cam_state.position = cam_poses[i].translation(); // t_c_w
    cam_states[new_cam_state.id] = new_cam_state;
  }

  // Compute measurements.
  random_numbers::RandomNumberGenerator noise_generator;
  vector<Vector4d, aligned_allocator<Vector4d> > measurements(num);
  vector<cv::Point2d> features;
  for (int i = 0; i < num; ++i) {
    Isometry3d cam_pose_inv = cam_poses[i].inverse();// R_w_c
    Vector3d p = cam_pose_inv.linear()*feature + cam_pose_inv.translation();//pc
    
    double u = p(0) / p(2) + noise_generator.gaussian(0.0, 0.001);
    double v = p(1) / p(2) + noise_generator.gaussian(0.0, 0.001);
    // double u = p(0) / p(2);
    // double v = p(1) / p(2);
    Vector3d pc = intrinsics0 * p;
    features.push_back(cv::Point2d(pc(0)/pc(2), pc(1)/pc(2)));
    
    measurements[i] = Vector4d(u, v, u, v);
  }

  // generate uv measurement
  vector<cv::Point2f> cam0_points_undistorted(0);
  vector<cv::Point2f> cam1_points_undistorted(0);

  vector<cv::Point2f> cam0_points;
  for (int i = 0; i < num; i++){
    cam0_points_undistorted.push_back(cv::Point2f(measurements[i](0), measurements[i](1)));
  }

  cam0_points = distortPoints(cam0_points_undistorted, cam0_intrinsics,
                          cam0_distortion_model, cam0_distortion_coeffs);

  
  
  Eigen::Vector4d feature_uv(cam0_points[0].x, cam0_points[0].y,
              cam0_points[1].x, cam0_points[1].y);

  for (int i = 0; i < num; i++){
    cout << "feature: " << features[i].x << " " <<  features[i].y << " measure: " << cam0_points[i].x << " " << cam0_points[i].y << endl;
  }


  // Initialize a feature object.
  Feature feature_object;
  for (int i = 0; i < num; ++i)
    feature_object.observations[i] = measurements[i];

  // Compute the 3d position of the feature.
  feature_object.initializePosition(cam_states);
  cout << "cam state size: " << cam_states.size() << endl;
  // Check the difference between the computed 3d
  // feature position and the groud truth.
  cout << "ground truth position: " << feature.transpose() << endl;
  cout << "estimated position: " << feature_object.position.transpose() << endl;
  Eigen::Vector3d error = feature_object.position - feature;

  // triangulate

  cv::Mat T0, T1, P0, P1;
  cv::Mat x3D;
  cv::Mat K0 = cv::Mat_<double>(3, 3), K1 = cv::Mat_<double>(3, 3);
  cv::eigen2cv(intrinsics0, K0);
  cv::eigen2cv(intrinsics1, K1);

  T0 = (cv::Mat_<double>(3, 4) << 
      1,0,0,0,
      0,1,0,0,
      0,0,1,0);
  P0 = K0 * T0;
  
  Isometry3d  T_cam0_cam1 = cam_poses[1].inverse() * cam_poses[0];
  
  T1 = (cv::Mat_<double>(3, 4) << 
      T_cam0_cam1.linear()(0,0), T_cam0_cam1.linear()(0,1), T_cam0_cam1.linear()(0,2), T_cam0_cam1.translation()[0],
      T_cam0_cam1.linear()(1,0), T_cam0_cam1.linear()(1,1), T_cam0_cam1.linear()(1,2), T_cam0_cam1.translation()[1],
      T_cam0_cam1.linear()(2,0), T_cam0_cam1.linear()(2,1), T_cam0_cam1.linear()(2,2), T_cam0_cam1.translation()[2]);
  P1 = K0 * T1;

  //orb-slam
  cv::Mat A = cv::Mat_<double>(4, 4);
  cv::Mat x;

  x = feature_uv(0) * P0.row(2)-P0.row(0);
  x.copyTo(A.row(0));

  x = feature_uv(1) * P0.row(2)-P0.row(1);
  x.copyTo(A.row(1));
  
  x = feature_uv(2) * P1.row(2)-P1.row(0);
  x.copyTo(A.row(2));

  x = feature_uv(3) * P1.row(2)-P1.row(1);
  x.copyTo(A.row(3));

  cv::Mat u,w,vt;

  cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0,3)/x3D.at<double>(3);

  Vector3d position(x3D.at<double>(0,0), x3D.at<double>(1,0), x3D.at<double>(2,0));
  cout << position.transpose() << endl;
  Vector3d pw = cam_poses[0] * position;
  cout << "triangulate: " << pw.transpose() << endl;
  EXPECT_NEAR(error.norm(), 0, 0.05);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

