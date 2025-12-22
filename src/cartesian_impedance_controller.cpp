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
//  ─────────────────────────────────────────────────────┐
// │  Cartesian Impedance Control Flow                   │
// └─────────────────────────────────────────────────────┘

// 1. Current EE pose:        x_current (from forward kinematics)
// 2. Desired EE pose:        x_desired (from topic)
// 3. Cartesian error:        e = x_current - x_desired  [6x1]
// 4. Cartesian velocity:     v = J · dq                 [6x1]
// 5. Cartesian force:        F = -K·e - D·v             [6x1]
// 6. Joint torques:          τ = J^T · F + coriolis    [7x1]
//                                   ↑
//                          Jacobian transpose maps 
//                          Cartesian forces to joint torques

#include <franka_iri_controllers/cartesian_impedance_controller.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <chrono>

#include <Eigen/Dense>

namespace franka_iri_controllers {

void CartesianImpedanceController::sendGripperGoal(bool close_gripper) {
  if (!is_gripper_loaded_) {
    RCLCPP_WARN(get_node()->get_logger(), "Gripper not loaded; ignoring /gripper_command.");
    return;
  }

  const auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(std::max(0.0, gripper_action_wait_timeout_s_)));

  if (close_gripper) {
    if (!gripper_grasp_action_client_) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gripper grasp action client not initialized.");
      return;
    }
    if (!gripper_grasp_action_client_->wait_for_action_server(timeout)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Gripper grasp action server not available at '%s'",
                   gripper_grasp_action_name_.c_str());
      return;
    }

    auto goal_msg = GripperGrasp::Goal();
    goal_msg.width = gripper_close_width_;
    goal_msg.speed = gripper_close_speed_;
    goal_msg.force = gripper_close_force_;
    goal_msg.epsilon.inner = gripper_close_epsilon_inner_;
    goal_msg.epsilon.outer = gripper_close_epsilon_outer_;

    auto send_goal_options =
        rclcpp_action::Client<GripperGrasp>::SendGoalOptions();
    send_goal_options.result_callback =
        [node = get_node()](
            const rclcpp_action::ClientGoalHandle<GripperGrasp>::WrappedResult& result) {
          if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(node->get_logger(), "Gripper CLOSE successful");
          } else {
            RCLCPP_WARN(node->get_logger(), "Gripper CLOSE failed");
          }
        };

    gripper_grasp_action_client_->async_send_goal(goal_msg, send_goal_options);
    return;
  }

  // Open gripper
  if (!gripper_move_action_client_) {
    RCLCPP_ERROR(get_node()->get_logger(), "Gripper move action client not initialized.");
    return;
  }
  if (!gripper_move_action_client_->wait_for_action_server(timeout)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Gripper move action server not available at '%s'",
                 gripper_move_action_name_.c_str());
    return;
  }

  auto goal_msg = GripperMove::Goal();
  goal_msg.width = gripper_open_width_;
  goal_msg.speed = gripper_open_speed_;

  auto send_goal_options = rclcpp_action::Client<GripperMove>::SendGoalOptions();
  send_goal_options.result_callback =
      [node = get_node()](
          const rclcpp_action::ClientGoalHandle<GripperMove>::WrappedResult& result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(node->get_logger(), "Gripper OPEN successful");
        } else {
          RCLCPP_WARN(node->get_logger(), "Gripper OPEN failed");
        }
      };

  gripper_move_action_client_->async_send_goal(goal_msg, send_goal_options);
}

void CartesianImpedanceController::gripperCommandCallback(
    const std_msgs::msg::Bool::SharedPtr msg) {
  const bool close_gripper = msg->data;
  if (have_last_gripper_command_ && close_gripper == last_gripper_close_command_) {
    return;
  }
  have_last_gripper_command_ = true;
  last_gripper_close_command_ = close_gripper;

  RCLCPP_INFO(get_node()->get_logger(), "Gripper command: %s", close_gripper ? "CLOSE" : "OPEN");
  sendGripperGoal(close_gripper);
}

controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Effort command interfaces for all joints
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Cartesian pose state interfaces
  config.names = franka_cartesian_pose_->get_state_interface_names();

  // Joint position and velocity state interfaces
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
  }
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }

  // Robot model state interfaces (for Jacobian, coriolis, etc.)
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }

  return config;
}

void CartesianImpedanceController::updateJointStates() {
  // Joint positions start after 16 cartesian pose state interfaces
  for (int i = 0; i < num_joints_; ++i) {
    const auto& position_interface = state_interfaces_.at(16 + i);
    const auto& velocity_interface = state_interfaces_.at(16 + num_joints_ + i);
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

void CartesianImpedanceController::deltaPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(delta_pose_mutex_);

  delta_position_ = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y,
                                    msg->pose.position.z);

  delta_orientation_ = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                          msg->pose.orientation.y, msg->pose.orientation.z);

  // Treat incoming orientation as a delta quaternion; ensure unit length.
  const double n = delta_orientation_.norm();
  if (std::isfinite(n) && n > 1e-12) {
    delta_orientation_.coeffs() /= n;
  } else {
    delta_orientation_ = Eigen::Quaterniond::Identity();
  }

  new_delta_received_ = true;
}

Eigen::Vector3d CartesianImpedanceController::computeOrientationError(
    const Eigen::Quaterniond& orientation_d,
    const Eigen::Quaterniond& orientation) {
  // Ensure quaternion scalar product is positive (shortest path)
  Eigen::Quaterniond orientation_corrected = orientation;
  if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation_corrected.coeffs() << -orientation.coeffs();
  }

  // "Difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d);
  error_quaternion.normalize();
  if (error_quaternion.w() < 0.0) {
    error_quaternion.coeffs() *= -1.0;
  }

  const Eigen::Vector3d v(error_quaternion.x(), error_quaternion.y(), error_quaternion.z());
  const double v_norm = v.norm();
  Eigen::Vector3d rot_vec = Eigen::Vector3d::Zero();
  if (v_norm > 1e-12) {
    const double angle = 2.0 * std::atan2(v_norm, error_quaternion.w());
    rot_vec = angle * (v / v_norm);
  }

  // Transform error to base frame and match libfranka sign convention.
  return -orientation_corrected.toRotationMatrix() * rot_vec;
}

controller_interface::return_type CartesianImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period/*period*/) {
  // Update joint states
  updateJointStates();

  // Get current Cartesian pose
  auto [current_orientation, current_position] =
      franka_cartesian_pose_->getCurrentOrientationAndTranslation();

  // Get Jacobian from model
  std::array<double, 42> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

  // Get Coriolis or Gravity compensation
  const std::array<double, 7> compensation_array =
      use_gravity_compensation_ ? franka_robot_model_->getGravityForceVector()
                                : franka_robot_model_->getCoriolisForceVector();
  Eigen::Map<const Vector7d> compensation(compensation_array.data());

  // First-order low-pass on measured joint velocities.
  dq_filter_alpha_ = std::clamp(dq_filter_alpha_, 0.0, 1.0);
  dq_filtered_ = (1.0 - dq_filter_alpha_) * dq_filtered_ + dq_filter_alpha_ * dq_;

  // Initialize on first update
  if (initialization_flag_) {
    initial_orientation_ = current_orientation;
    initial_position_ = current_position;
    desired_orientation_ = current_orientation;
    desired_position_ = current_position;
    delta_orientation_ = Eigen::Quaterniond::Identity();
    delta_position_.setZero();
    dq_filtered_.setZero();
    tau_commanded_ = compensation;

    // Nullspace target is the startup joint configuration.
    q_nullspace_target_ = q_;
    initialization_flag_ = false;
  }

  // Get desired pose (thread-safe)
  // Only update when new delta is received, otherwise maintain previous desired pose
  Eigen::Vector3d position_d;
  Eigen::Quaterniond orientation_d;
  {
    std::lock_guard<std::mutex> lock(delta_pose_mutex_);
    if (new_delta_received_) {
      Eigen::Vector3d target_position = current_position + delta_position_;
      
      // Clamp maximum position error to prevent large jumps
      const double max_position_error = std::max(0.0, delta_pose_max_position_error_);
      Eigen::Vector3d error_vec = target_position - current_position;
      double error_norm = error_vec.norm();
      if (max_position_error > 0.0 && error_norm > max_position_error) {
        target_position = current_position + error_vec * (max_position_error / error_norm);
      }
      
      // Smooth transition: blend towards target instead of instant jump
      const double alpha = std::clamp(delta_pose_alpha_, 0.0, 1.0);
      desired_position_ = desired_position_ * (1.0 - alpha) + target_position * alpha;

      // Quaternion-only orientation update (no Euler). Use slerp to avoid step changes.
      Eigen::Quaterniond target_orientation = delta_orientation_ * current_orientation;
      target_orientation.normalize();
      desired_orientation_ = desired_orientation_.slerp(alpha, target_orientation);
      desired_orientation_.normalize();
      new_delta_received_ = false;
    }
    position_d = desired_position_;
    orientation_d = desired_orientation_;
  }
  // Compute Cartesian error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << current_position - position_d;
  error.tail(3) << computeOrientationError(orientation_d, current_orientation);

  // Optional per-axis error clipping (negative values disable clipping for that direction).
  for (int axis = 0; axis < 3; ++axis) {
    const double pos_clip = trans_clip_pos_(axis);
    const double neg_clip = trans_clip_neg_(axis);
    if (pos_clip >= 0.0) {
      error(axis) = std::min(error(axis), pos_clip);
    }
    if (neg_clip >= 0.0) {
      error(axis) = std::max(error(axis), -neg_clip);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const int idx = 3 + axis;
    const double pos_clip = rot_clip_pos_(axis);
    const double neg_clip = rot_clip_neg_(axis);
    if (pos_clip >= 0.0) {
      error(idx) = std::min(error(idx), pos_clip);
    }
    if (neg_clip >= 0.0) {
      error(idx) = std::max(error(idx), -neg_clip);
    }
  }
  
  // Print error periodically (every ~100 updates to avoid spam)
  static int counter = 0;
  if (++counter % 100 == 0) {
    RCLCPP_INFO(get_node()->get_logger(), "Position error: [%.4f, %.4f, %.4f], Orientation error: [%.4f, %.4f, %.4f]", 
                error(0), error(1), error(2), error(3), error(4), error(5));
  }
  // Compute Cartesian velocity
  Eigen::Matrix<double, 6, 1> velocity = jacobian * dq_filtered_;

  // PD Impedance control law: tau = J^T * (-K .* error - D .* velocity) + coriolis
  // Use element-wise multiplication (cwiseProduct) like joint impedance controller
  Eigen::Matrix<double, 6, 1> force = -k_gains_.cwiseProduct(error) - d_gains_.cwiseProduct(velocity);
  Vector7d tau_d = jacobian.transpose() * force + compensation;

  // Optional nullspace posture control (projected to not affect the Cartesian task).
  if (nullspace_stiffness_.maxCoeff() > 0.0 || nullspace_damping_ > 0.0) {
    Eigen::Matrix<double, 6, 6> JJt = jacobian * jacobian.transpose();
    JJt.diagonal().array() += nullspace_projection_damping_;
    const Eigen::Matrix<double, 6, 6> JJt_inv =
        JJt.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
    const Eigen::Matrix<double, 7, 6> J_pinv = jacobian.transpose() * JJt_inv;
    const Eigen::Matrix<double, 7, 7> N =
        Eigen::Matrix<double, 7, 7>::Identity() - J_pinv * jacobian;

    const Vector7d tau_ns = nullspace_stiffness_.cwiseProduct(q_nullspace_target_ - q_) -
                            nullspace_damping_ * dq_filtered_;
    tau_d += N * tau_ns;
  }

  // Apply torque rate limiting to prevent velocity violations
  // Torque-rate limit must scale with controller update period.
    // The previous hard-coded 1 ms dt effectively crippled the controller
    // if the actual update rate was lower (e.g., 100-500 Hz in Gazebo).
  const double max_torque_rate = std::max(0.0, max_torque_rate_);  // [Nm/s]
  const double dt = std::max(period.seconds(), 1e-4);
  const double delta_tau_max = max_torque_rate * dt;
  
  for (int i = 0; i < num_joints_; ++i) {
    double delta_tau = tau_d(i) - tau_commanded_(i);
    delta_tau = std::clamp(delta_tau, -delta_tau_max, delta_tau_max);
    tau_commanded_(i) += delta_tau;
  }

  // Send rate-limited torque commands
  for (int i = 0; i < num_joints_; ++i) {
    command_interfaces_[i].set_value(tau_commanded_(i));
  }

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>("k_gains", {50.0, 50.0, 50.0, 5.0, 5.0, 5.0});
    auto_declare<std::vector<double>>("d_gains", {3.0, 3.0, 3.0, 2.0, 2.0, 2.0});
    auto_declare<bool>("load_gripper", true);

    auto_declare<bool>("use_gravity_compensation", false);
    auto_declare<double>("dq_filter_alpha", 0.5);
    auto_declare<double>("max_torque_rate", 50.0);

    // Delta pose smoothing / safety
    auto_declare<double>("delta_pose_alpha", 0.3);
    auto_declare<double>("delta_pose_max_position_error", 0.3);

    // Optional Cartesian error clipping (negative values disable clipping).
    auto_declare<double>("translational_clip_x", -1.0);
    auto_declare<double>("translational_clip_y", -1.0);
    auto_declare<double>("translational_clip_z", -1.0);
    auto_declare<double>("translational_clip_neg_x", -1.0);
    auto_declare<double>("translational_clip_neg_y", -1.0);
    auto_declare<double>("translational_clip_neg_z", -1.0);
    auto_declare<double>("rotational_clip_x", -1.0);
    auto_declare<double>("rotational_clip_y", -1.0);
    auto_declare<double>("rotational_clip_z", -1.0);
    auto_declare<double>("rotational_clip_neg_x", -1.0);
    auto_declare<double>("rotational_clip_neg_y", -1.0);
    auto_declare<double>("rotational_clip_neg_z", -1.0);

    // Optional nullspace posture control (0 disables).
    auto_declare<double>("nullspace_stiffness", 0.0);
    auto_declare<double>("joint1_nullspace_stiffness", -1.0);
    auto_declare<double>("nullspace_damping", 0.0);
    auto_declare<double>("nullspace_projection_damping", 1e-6);

    // Gripper command interface (true=close, false=open)
    auto_declare<std::string>("gripper_command_topic", "/gripper_command");

    // Franka gripper actions
    auto_declare<std::string>("gripper_grasp_action_name", "franka_gripper/grasp");
    auto_declare<std::string>("gripper_move_action_name", "franka_gripper/move");

    // Open goal (Move)
    auto_declare<double>("gripper_open_width", 0.08);
    auto_declare<double>("gripper_open_speed", 0.05);

    // Close goal (Grasp)
    auto_declare<double>("gripper_close_width", 0.0);
    auto_declare<double>("gripper_close_speed", 0.05);
    auto_declare<double>("gripper_close_force", 60.0);
    auto_declare<double>("gripper_close_epsilon_inner", 0.005);
    auto_declare<double>("gripper_close_epsilon_outer", 0.005);

    auto_declare<double>("gripper_action_wait_timeout_s", 1.0);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during on_init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  // Initialize Cartesian pose interface (no elbow control)
  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(false);

  return CallbackReturn::SUCCESS;
}

bool CartesianImpedanceController::assign_parameters() {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  is_gripper_loaded_ = get_node()->get_parameter("load_gripper").as_bool();

  use_gravity_compensation_ = get_node()->get_parameter("use_gravity_compensation").as_bool();
  dq_filter_alpha_ = get_node()->get_parameter("dq_filter_alpha").as_double();
  max_torque_rate_ = get_node()->get_parameter("max_torque_rate").as_double();
  delta_pose_alpha_ = get_node()->get_parameter("delta_pose_alpha").as_double();
  delta_pose_max_position_error_ =
      get_node()->get_parameter("delta_pose_max_position_error").as_double();

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  
  // For Cartesian impedance, we need 6 gains (x, y, z, rx, ry, rz)
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return false;
  }
  if (k_gains.size() != 6) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size 6 but is of size %ld",
                 k_gains.size());
    return false;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return false;
  }
  if (d_gains.size() != 6) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size 6 but is of size %ld",
                 d_gains.size());
    return false;
  }
  
  for (int i = 0; i < 6; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }

    // Error clipping (negative values mean disabled for that direction)
    trans_clip_pos_ << get_node()->get_parameter("translational_clip_x").as_double(),
      get_node()->get_parameter("translational_clip_y").as_double(),
      get_node()->get_parameter("translational_clip_z").as_double();
    trans_clip_neg_ << get_node()->get_parameter("translational_clip_neg_x").as_double(),
      get_node()->get_parameter("translational_clip_neg_y").as_double(),
      get_node()->get_parameter("translational_clip_neg_z").as_double();

    rot_clip_pos_ << get_node()->get_parameter("rotational_clip_x").as_double(),
      get_node()->get_parameter("rotational_clip_y").as_double(),
      get_node()->get_parameter("rotational_clip_z").as_double();
    rot_clip_neg_ << get_node()->get_parameter("rotational_clip_neg_x").as_double(),
      get_node()->get_parameter("rotational_clip_neg_y").as_double(),
      get_node()->get_parameter("rotational_clip_neg_z").as_double();

    // Nullspace posture control
    const double nullspace_stiffness_scalar =
      std::max(0.0, get_node()->get_parameter("nullspace_stiffness").as_double());
    nullspace_stiffness_.setConstant(nullspace_stiffness_scalar);
    const double joint1_ns = get_node()->get_parameter("joint1_nullspace_stiffness").as_double();
    if (joint1_ns >= 0.0) {
    nullspace_stiffness_(0) = joint1_ns;
    }
    nullspace_damping_ = std::max(0.0, get_node()->get_parameter("nullspace_damping").as_double());
    nullspace_projection_damping_ =
      std::max(1e-12, get_node()->get_parameter("nullspace_projection_damping").as_double());

      gripper_command_topic_ = get_node()->get_parameter("gripper_command_topic").as_string();

      gripper_grasp_action_name_ =
        get_node()->get_parameter("gripper_grasp_action_name").as_string();
      gripper_move_action_name_ =
        get_node()->get_parameter("gripper_move_action_name").as_string();

      gripper_open_width_ = get_node()->get_parameter("gripper_open_width").as_double();
      gripper_open_speed_ = get_node()->get_parameter("gripper_open_speed").as_double();

      gripper_close_width_ = get_node()->get_parameter("gripper_close_width").as_double();
      gripper_close_speed_ = get_node()->get_parameter("gripper_close_speed").as_double();
      gripper_close_force_ = get_node()->get_parameter("gripper_close_force").as_double();
      gripper_close_epsilon_inner_ =
        get_node()->get_parameter("gripper_close_epsilon_inner").as_double();
      gripper_close_epsilon_outer_ =
        get_node()->get_parameter("gripper_close_epsilon_outer").as_double();

      gripper_action_wait_timeout_s_ =
        get_node()->get_parameter("gripper_action_wait_timeout_s").as_double();
  
  RCLCPP_INFO(get_node()->get_logger(), "K gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
              k_gains_(0), k_gains_(1), k_gains_(2), k_gains_(3), k_gains_(4), k_gains_(5));
  RCLCPP_INFO(get_node()->get_logger(), "D gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
              d_gains_(0), d_gains_(1), d_gains_(2), d_gains_(3), d_gains_(4), d_gains_(5));
  
  return true;
}

CallbackReturn CartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!assign_parameters()) {
    return CallbackReturn::FAILURE;
  }

  // Initialize robot model interface
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name_,
                                                   arm_id_ + "/" + k_robot_state_interface_name_));

  // Set default collision behavior
  auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");
  auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();

  auto future_result = client->async_send_request(request);
  future_result.wait_for(robot_utils::time_out);

  auto success = future_result.get();
  if (!success) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
    return CallbackReturn::ERROR;
  }
  RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");

  // Get robot description and arm_id from robot_state_publisher
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

  // Create delta_pose subscriber
  delta_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/delta_pose", rclcpp::QoS(1).best_effort(),
      std::bind(&CartesianImpedanceController::deltaPoseCallback, this, std::placeholders::_1));

    // Create gripper action clients + command subscriber
    gripper_grasp_action_client_ =
      rclcpp_action::create_client<GripperGrasp>(get_node(), gripper_grasp_action_name_);
    gripper_move_action_client_ =
      rclcpp_action::create_client<GripperMove>(get_node(), gripper_move_action_name_);
    gripper_command_sub_ = get_node()->create_subscription<std_msgs::msg::Bool>(
      gripper_command_topic_, rclcpp::QoS(10).reliable(),
      std::bind(&CartesianImpedanceController::gripperCommandCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(),
              "Cartesian Impedance Controller configured. Subscribing to /delta_pose");

  RCLCPP_INFO(get_node()->get_logger(),
              "Gripper control enabled: topic '%s' -> grasp '%s' (close) / move '%s' (open)",
              gripper_command_topic_.c_str(), gripper_grasp_action_name_.c_str(),
              gripper_move_action_name_.c_str());

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  initialization_flag_ = true;
  dq_filtered_.setZero();

  // Assign state interfaces to semantic components
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  RCLCPP_INFO(get_node()->get_logger(), "Cartesian Impedance Controller activated.");

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->release_interfaces();
  franka_robot_model_->release_interfaces();

  return CallbackReturn::SUCCESS;
}

}  // namespace franka_iri_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_iri_controllers::CartesianImpedanceController,
                       controller_interface::ControllerInterface)
