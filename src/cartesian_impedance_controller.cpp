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

#include <franka_iri_controllers/cartesian_impedance_controller.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include <Eigen/Dense>

namespace franka_iri_controllers {

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
  // RCLCPP_INFO(get_node()->get_logger(), "Received new delta pose.");
  // RCLCPP_INFO(get_node()->get_logger(), "Delta position: [%f, %f, %f]", delta_position_.x(),
  //             delta_position_.y(), delta_position_.z());
  // RCLCPP_INFO(get_node()->get_logger(), "Delta orientation (quat): [%f, %f, %f, %f]",
  //             delta_orientation_.w(), delta_orientation_.x(), delta_orientation_.y(),
  //             delta_orientation_.z());
  // Compute new desired pose = initial + delta
  desired_position_ = initial_position_ + delta_position_;
  desired_orientation_ = delta_orientation_ * initial_orientation_;
  desired_orientation_.normalize();
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

  // Extract vector part and transform to base frame
  Eigen::Vector3d error;
  error << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

  // Transform error to base frame
  return orientation_corrected.toRotationMatrix() * error;
}

controller_interface::return_type CartesianImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  // Update joint states
  updateJointStates();

  // Get current Cartesian pose
  auto [current_orientation, current_position] =
      franka_cartesian_pose_->getCurrentOrientationAndTranslation();

  // Initialize on first update
  if (initialization_flag_) {
    initial_orientation_ = current_orientation;
    initial_position_ = current_position;
    desired_orientation_ = current_orientation;
    desired_position_ = current_position;
    delta_orientation_ = Eigen::Quaterniond::Identity();
    delta_position_.setZero();
    dq_filtered_.setZero();
    tau_commanded_.setZero();
    initialization_flag_ = false;
  }

  // Get desired pose (thread-safe)
  Eigen::Vector3d position_d;
  Eigen::Quaterniond orientation_d;
  {
    std::lock_guard<std::mutex> lock(delta_pose_mutex_);
    position_d = desired_position_;
    orientation_d = desired_orientation_;
  }
  // Print desired pose for debugging
  // RCLCPP_INFO(get_node()->get_logger(), "Desired position: [%f, %f, %f]", position_d.x(), position_d.y(),
  //              position_d.z());
  // RCLCPP_INFO(get_node()->get_logger(), "Desired orientation (quat): [%f, %f, %f, %f]",
  //              orientation_d.w(), orientation_d.x(), orientation_d.y(), orientation_d.z());

  // Compute Cartesian error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << current_position - position_d;
  // error.tail(3) << computeOrientationError(orientation_d, current_orientation);
  // Print error
  RCLCPP_INFO(get_node()->get_logger(), "Position error: [%f, %f, %f]", error(0), error(1),
               error(2));
  // Get Jacobian from model
  std::array<double, 42> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

  // Get Coriolis forces
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  Eigen::Map<const Vector7d> coriolis(coriolis_array.data());

  // Filter joint velocities (same as other working controllers)
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // Compute Cartesian velocity
  Eigen::Matrix<double, 6, 1> velocity = jacobian * dq_filtered_;

  // Impedance control law: tau = J^T * (-K * error - D * velocity) + coriolis
  // This is the standard Cartesian impedance control formula
  Vector7d tau_d = jacobian.transpose() * (-stiffness_ * error - damping_ * velocity) + coriolis;

  // Send torque commands directly (same pattern as joint_impedance_with_ik_example_controller)
  for (int i = 0; i < num_joints_; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<double>("translational_stiffness", 300.0);
    auto_declare<double>("rotational_stiffness", 30.0);
    auto_declare<double>("nullspace_stiffness", 0.5);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during on_init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  // Initialize Cartesian pose interface (no elbow control)
  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(false);

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Get parameters
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  translational_stiffness_ = get_node()->get_parameter("translational_stiffness").as_double();
  rotational_stiffness_ = get_node()->get_parameter("rotational_stiffness").as_double();
  nullspace_stiffness_ = get_node()->get_parameter("nullspace_stiffness").as_double();

  // Setup stiffness and damping matrices
  stiffness_.setZero();
  stiffness_.topLeftCorner(3, 3) << translational_stiffness_ * Eigen::Matrix3d::Identity();
  stiffness_.bottomRightCorner(3, 3) << rotational_stiffness_ * Eigen::Matrix3d::Identity();

  damping_.setZero();
  damping_.topLeftCorner(3, 3) << 2.0 * std::sqrt(translational_stiffness_) *
                                      Eigen::Matrix3d::Identity();
  damping_.bottomRightCorner(3, 3) << 2.0 * std::sqrt(rotational_stiffness_) *
                                          Eigen::Matrix3d::Identity();

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

  RCLCPP_INFO(get_node()->get_logger(),
              "Cartesian Impedance Controller configured. Subscribing to /delta_pose");

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

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_iri_controllers::CartesianImpedanceController,
                       controller_interface::ControllerInterface)
