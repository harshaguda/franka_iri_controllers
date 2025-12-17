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
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/bool.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>

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
  using GripperGrasp = franka_msgs::action::Grasp;
  using GripperMove = franka_msgs::action::Move;

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

  /**
   * @brief Assign parameters from ROS2 parameter server
   */
  bool assign_parameters();

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
  bool new_delta_received_{false};

  // Joint states
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

  // Impedance control gains (PD control in Cartesian space)
  // k_gains: stiffness for [x, y, z, rx, ry, rz] (6 DOF Cartesian space)
  // d_gains: damping for [x, y, z, rx, ry, rz]
  Eigen::Matrix<double, 6, 1> k_gains_;
  Eigen::Matrix<double, 6, 1> d_gains_;
  bool is_gripper_loaded_ = true;

  // Torque filtering and limiting
  Vector7d tau_commanded_;
    double dq_filter_alpha_{0.5};
    double max_torque_rate_{50.0};  // [Nm/s]

    bool use_gravity_compensation_{false};

    // Delta-pose smoothing and safety
    double delta_pose_alpha_{0.3};
    double delta_pose_max_position_error_{0.3};

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

  // Gripper control
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr gripper_command_sub_;
  rclcpp_action::Client<GripperGrasp>::SharedPtr gripper_grasp_action_client_;
  rclcpp_action::Client<GripperMove>::SharedPtr gripper_move_action_client_;
  bool last_gripper_close_command_{false};
  bool have_last_gripper_command_{false};

  std::string gripper_command_topic_{"/gripper_command"};
  std::string gripper_grasp_action_name_{"franka_gripper/grasp"};
  std::string gripper_move_action_name_{"franka_gripper/move"};

  double gripper_open_width_{0.08};
  double gripper_open_speed_{0.05};

  double gripper_close_width_{0.0};
  double gripper_close_speed_{0.05};
  double gripper_close_force_{60.0};
  double gripper_close_epsilon_inner_{0.005};
  double gripper_close_epsilon_outer_{0.005};

  double gripper_action_wait_timeout_s_{1.0};

  void gripperCommandCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void sendGripperGoal(bool close_gripper);
};

}  // namespace franka_iri_controllers
