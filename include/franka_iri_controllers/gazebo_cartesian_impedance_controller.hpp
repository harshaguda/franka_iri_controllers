// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include <franka_example_controllers/robot_utils.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/se3.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_iri_controllers {

class GazeboCartesianImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
	  const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
	  const override;
  controller_interface::return_type update(const rclcpp::Time& time,
										   const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  void updateJointStates();
  void deltaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  Eigen::Vector3d computeOrientationError(const Eigen::Quaterniond& orientation_d,
										  const Eigen::Quaterniond& orientation);
  bool assign_parameters();

  bool build_pinocchio_model_from_urdf(const std::string& urdf_xml);
  bool initialize_pinocchio_joint_and_frame_mapping();
  void compute_current_pose_and_jacobian(Eigen::Quaterniond& current_orientation,
										Eigen::Vector3d& current_position,
										Eigen::Matrix<double, 6, 7>& jacobian);
  Vector7d compute_coriolis_and_gravity_torque();

  Eigen::Quaterniond initial_orientation_;
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond desired_orientation_;
  Eigen::Vector3d desired_position_;

  Eigen::Quaterniond delta_orientation_;
  Eigen::Vector3d delta_position_;
  std::mutex delta_pose_mutex_;
  bool new_delta_received_{false};

  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;

  // Nullspace posture control (holds the startup joint configuration)
  Vector7d q_nullspace_target_{Vector7d::Zero()};
  Vector7d nullspace_stiffness_{Vector7d::Zero()};
  double nullspace_damping_{0.0};
  double nullspace_projection_damping_{1e-6};

  // Cartesian error clipping (per-axis, symmetrical via separate +/- values).
  // Negative values disable clipping for that direction.
  Eigen::Vector3d trans_clip_pos_{Eigen::Vector3d::Constant(-1.0)};
  Eigen::Vector3d trans_clip_neg_{Eigen::Vector3d::Constant(-1.0)};
  Eigen::Vector3d rot_clip_pos_{Eigen::Vector3d::Constant(-1.0)};
  Eigen::Vector3d rot_clip_neg_{Eigen::Vector3d::Constant(-1.0)};

  Eigen::Matrix<double, 6, 1> k_gains_;
  Eigen::Matrix<double, 6, 1> d_gains_;
  bool is_gripper_loaded_ = true;

  Vector7d tau_commanded_;

  bool initialization_flag_{true};
  std::string arm_id_{"fr3"};
  std::string robot_description_;
  static constexpr int num_joints_{7};

  bool pinocchio_ready_{false};
  bool use_gravity_compensation_{true};
  std::string ee_frame_{""};
  pinocchio::Model model_;
  pinocchio::Data data_{model_};
  pinocchio::FrameIndex ee_frame_id_{0};
  std::array<pinocchio::JointIndex, num_joints_> joint_ids_{};
  std::array<int, num_joints_> joint_q_indices_{};
  std::array<int, num_joints_> joint_v_indices_{};

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr delta_pose_sub_;
};

}  // namespace franka_iri_controllers
