/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_FEATURE_H
#define MSCKF_VIO_FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"

#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

using namespace std;

namespace msckf_vio {

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int FeatureIDType;

  /*
   * @brief OptimizationConfig Configuration parameters
   *    for 3d feature position optimization.
   */
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @param cam_states : input camera poses.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion(
      const CamStateServer& cam_states) const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @param cam_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition(
      const CamStateServer& cam_states);


  // An unique identifier for the feature.
  // In case of long time running, the variable
  // type of id is set to FeatureIDType in order
  // to avoid duplication.
  FeatureIDType id;

  // id for next feature
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector4d> > > observations;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

  // Noise for a normalized feature measurement.
  static double observation_noise;
  static Eigen::Matrix<double, 3, 4> K0;
  

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;


void Feature::cost(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);

  return;
}

void Feature::generateInitialGuess(
    const Eigen::Isometry3d& T_c_c0, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  
  cv::Matx33d K0(458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
  // 457.587, 456.134, 379.999, 255.238
  cv::Matx33d K1(457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0);
  std::vector<cv::Point2d> pts_in, pts_out;
  pts_in.push_back(cv::Point2d(z1[0], z1[1]));
  pts_in.push_back(cv::Point2d(z2[0], z2[1]));

  cv::Vec4d distortion_coeffs(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

  std::vector<cv::Point3d> homogenous_pts;
  cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
  cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K0,
                    distortion_coeffs, pts_out);
  
  cv::Point2d c0_point, c_point;
  c0_point = pts_out[0];
  c_point = pts_out[0];

  cv::Mat T0, T1, P1, P2;
  T0 = (cv::Mat_<double>(3, 4) << 
        1,0,0,0,
        0,1,0,0,
        0,0,1,0); 
  cv::Mat x3D;
  P1 = K0 * T0;
  Eigen::Isometry3d T_12 = T_c_c0.inverse();
  T1 = (cv::Mat_<double>(3, 4) <<
      T_12.linear()(0,0), T_12.linear()(0,1), T_12.linear()(0,2), T_12.translation()(0),
      T_12.linear()(1,0), T_12.linear()(1,1), T_12.linear()(1,2), T_12.translation()(1),
      T_12.linear()(2,0), T_12.linear()(2,1), T_12.linear()(2,2), T_12.translation()(2));
  
  cv::Matx34d T_1_matx((double*)T1.ptr());
  P2 = K1 * T1;

  //orb-slam
  cv::Mat A = cv::Mat_<double>(4, 4);
  cv::Mat x;
  x = c0_point.x*P1.row(2)-P1.row(0);
  x.copyTo(A.row(0));

  x = c0_point.y*P1.row(2)-P1.row(1);
  x.copyTo(A.row(1));
  
  x = c_point.x*P2.row(2)-P2.row(0);
  x.copyTo(A.row(2));

  x = c_point.y*P2.row(2)-P2.row(1);
  x.copyTo(A.row(3));

  // std::cout << "A:" << A << std::endl;
  cv::Mat u,w,vt;

  cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0,3)/x3D.at<double>(3);
  // std::cout << "x3D:" << x3D << std::endl;

  p[0] = x3D.at<double>(0,0);
  p[1] = x3D.at<double>(1,0);
  p[2] = x3D.at<double>(2,0);
  // std::cout << "p:" << p << std::endl;

  // project error
  cv::Matx41d pw;
  cv::Matx31d u0, u1, p0, p1;
  cv::Matx31d zero;
  zero << 0,0,0;
  cv::Matx34d K0_extent, K1_extent;

  cv::hconcat(K0, zero, K0_extent);
  cv::hconcat(K1, zero, K1_extent);
  
  cv::Point2d error_0, error_1;
  
  pw << p[0], p[1], p[2], 1.;
  u0 = K0_extent * pw;
  p0(0,0) = u0(0,0)/u0(2,0);
  p0(1,0) = u0(1,0)/u0(2,0);
  error_0 = c0_point - cv::Point2d(p0(0,0),p0(1,0));
  // std::cout << "p0:" << p0 << std::endl;

  u1 = K1 * T_1_matx * pw;
  p1(0,0) = u1(0,0)/u1(2,0);
  p1(1,0) = u1(1,0)/u1(2,0);
  error_1 = c_point - cv::Point2d(p1(0,0),p1(1,0));
  // std::cout << "p1:" << p1 << std::endl;

  float rmse_0 = 0, rmse_1 = 0;
  rmse_0 += (error_0.x*error_0.x + error_0.y*error_0.y);
  
  rmse_1 += (error_1.x*error_1.x + error_1.y*error_1.y);
      

  rmse_0 = std::sqrt(rmse_0);
  rmse_1 = std::sqrt(rmse_1);
  cout << "rmse0:" << rmse_0 << endl;
  cout << "rmse1:" << rmse_1 << endl;


  // Construct a least square problem to solve the depth.
  // Eigen::Vector3d m = T_c_c0.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  // Eigen::Vector2d A(0.0, 0.0);
  // A(0) = m(0) - z2(0)*m(2);
  // A(1) = m(1) - z2(1)*m(2);

  // Eigen::Vector2d b(0.0, 0.0);
  // b(0) = z2(0)*T_c_c0.translation()(2) - T_c_c0.translation()(0);
  // b(1) = z2(1)*T_c_c0.translation()(2) - T_c_c0.translation()(1);

  // // Solve for the depth.
  // double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  // p(0) = z1(0) * depth;
  // p(1) = z1(1) * depth;
  // p(2) = depth;
  return;
}

bool Feature::checkMotion(
    const CamStateServer& cam_states) const {

  const StateIDType& first_cam_id = observations.begin()->first;
  const StateIDType& last_cam_id = (--observations.end())->first;

  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = quaternionToRotation(
      cam_states.find(first_cam_id)->second.orientation).transpose();
  first_cam_pose.translation() =
    cam_states.find(first_cam_id)->second.position;

  Eigen::Isometry3d last_cam_pose;
  last_cam_pose.linear() = quaternionToRotation(
      cam_states.find(last_cam_id)->second.orientation).transpose();
  last_cam_pose.translation() =
    cam_states.find(last_cam_id)->second.position;

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction(
      observations.begin()->second(0),
      observations.begin()->second(1), 1.0);
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear()*feature_direction;

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  Eigen::Vector3d translation = last_cam_pose.translation() -
    first_cam_pose.translation();
  double parallel_translation =
    translation.transpose()*feature_direction;
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;

  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)
    return true;
  else return false;
}

bool Feature::initializePosition(
    const CamStateServer& cam_states) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  for (auto& m : observations) {
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input cam_states buffer.
    auto cam_state_iter = cam_states.find(m.first);
    if (cam_state_iter == cam_states.end()) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());
    measurements.push_back(m.second.tail<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam0_pose;
    cam0_pose.linear() = quaternionToRotation(
        cam_state_iter->second.orientation).transpose();
    cam0_pose.translation() = cam_state_iter->second.position;

    Eigen::Isometry3d cam1_pose;
    cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();

    cam_poses.push_back(cam0_pose);
    cam_poses.push_back(cam1_pose);
  }

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;// T_w_c * T_c0_w = T_c_c0

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
      measurements[measurements.size()-1], initial_position);

  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));  // inverse depth

  // Eigen::Vector3d solution = initial_position;

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      jacobian(cam_poses[i], solution, measurements[i], J, r, w);

      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (
      delta_norm > optimization_config.estimation_precision);
    // }while (outer_loop_cntr++ <
    //   optimization_config.outer_loop_max_iteration &&
    //   delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  // Eigen::Vector3d final_position(solution(0)/solution(2),
  //     solution(1)/solution(2), 1.0/solution(2));

  Eigen::Vector3d final_position = initial_position;

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d position =
      pose.linear()*final_position + pose.translation();
    if (position(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }
  

  Eigen::Vector4d pw(final_position[0], final_position[1], final_position[2], 1.);
  Eigen::Matrix<double, 3, 4> K;
  K << 458.654, 0.0, 367.215,0, 0.0, 457.296, 248.375, 0, 0.0, 0.0, 1.0, 0;
  cv::Matx33d K0(458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
  Eigen::Vector3d u0 = K * pw;
  Eigen::Vector2d p0;
  p0(0) = u0(0)/u0(2);
  p0(1) = u0(1)/u0(2);

  std::vector<cv::Point2d> pts_out;
  cv::Point2d p(measurements[0][0], measurements[0][1]), error;
  std::vector<cv::Point2d> pts_in;
  pts_in.push_back(p);
  cv::Vec4d distortion_coeffs(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

  std::vector<cv::Point3d> homogenous_pts;

  cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
  cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K0,
                    distortion_coeffs, pts_out);
  error = pts_out[0] - cv::Point2d(p0(0), p0(1));
  std::cout << "error: " << error << std::endl;
  // Convert the feature position to the world frame.
  position = T_c0_w.linear()*final_position + T_c0_w.translation();

  if (is_valid_solution)
    is_initialized = true;

  std::cout << "is_valid_solution: " << is_valid_solution << std::endl;

  return is_valid_solution;
}
} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_H