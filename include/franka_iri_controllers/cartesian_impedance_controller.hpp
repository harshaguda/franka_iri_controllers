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
#include <memory>
#include <mutex>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include <franka_example_controllers/robot_utils.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_iri_controllers {

/**
 * @brief Cartesian impedance controller that subscribes to /delta_pose topic
 * 
 * This controller implements a Cartesian impedance control law. The equilibrium
 * point is set by subscribing to a /delta_pose topic, which provides delta
 * position/orientation that is added to the initial pose.
 */
class CartesianImpedanceController : public controller_interface::ControllerInterface {
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
  /**
   * @brief Update joint states from state interfaces
   */
  void updateJointStates();

  /**
   * @brief Callback for delta_pose subscription
   */
  void deltaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  /**
   * @brief Compute the orientation error between current and desired orientation
   */
  Eigen::Vector3d computeOrientationError(const Eigen::Quaterniond& orientation_d,
                                           const Eigen::Quaterniond& orientation);

  // Semantic interfaces
  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  // Initial and desired pose
  Eigen::Quaterniond initial_orientation_;
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond desired_orientation_;
  Eigen::Vector3d desired_position_;

  // Delta pose (from topic)
  Eigen::Quaterniond delta_orientation_;
  Eigen::Vector3d delta_position_;
  std::mutex delta_pose_mutex_;

  // Joint states
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;

  // Impedance parameters
  Eigen::Matrix<double, 6, 6> stiffness_;
  Eigen::Matrix<double, 6, 6> damping_;
  double translational_stiffness_{150.0};
  double rotational_stiffness_{10.0};
  double nullspace_stiffness_{0.5};

  // Torque filtering and limiting
  Vector7d tau_commanded_;
  static constexpr double kMaxTorqueRate{100.0};  // Nm/s - conservative limit
  static constexpr double kTorqueRateAlpha{0.9};  // Exponential filter: new = 0.1*target + 0.9*old

  // Controller state
  bool initialization_flag_{true};
  std::string arm_id_;
  std::string robot_description_;
  static constexpr int num_joints_{7};

  // Interface names
  const std::string k_robot_state_interface_name_{"robot_state"};
  const std::string k_robot_model_interface_name_{"robot_model"};

  // ROS subscriber
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr delta_pose_sub_;
};

}  // namespace franka_example_controllers
