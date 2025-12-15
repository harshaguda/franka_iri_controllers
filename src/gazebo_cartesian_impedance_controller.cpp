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
//
// 1. Current EE pose:        x_current (from forward kinematics)
// 2. Desired EE pose:        x_desired (from topic)
// 3. Cartesian error:        e = x_current - x_desired  [6x1]
// 4. Cartesian velocity:     v = J · dq                 [6x1]
// 5. Cartesian force:        F = -K·e - D·v             [6x1]
// 6. Joint torques:          τ = J^T · F + nle          [7x1]

#include "franka_iri_controllers/gazebo_cartesian_impedance_controller.hpp"

#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <memory>

#include <pluginlib/class_list_macros.hpp>

namespace franka_iri_controllers {

bool GazeboCartesianImpedanceController::build_pinocchio_model_from_urdf(const std::string& urdf_xml) {
    try {
        model_ = pinocchio::Model();
        pinocchio::urdf::buildModelFromXML(urdf_xml, model_);
        data_ = pinocchio::Data(model_);
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Failed to build Pinocchio model: %s", e.what());
        return false;
    }
}

bool GazeboCartesianImpedanceController::initialize_pinocchio_joint_and_frame_mapping() {
    ee_frame_ = get_node()->get_parameter("ee_frame").as_string();
    use_gravity_compensation_ = get_node()->get_parameter("use_gravity_compensation").as_bool();

    std::vector<std::string> candidates;
    if (!ee_frame_.empty()) {
        candidates.push_back(ee_frame_);
    }
    candidates.push_back(arm_id_ + "_hand_tcp");
    candidates.push_back(arm_id_ + "_hand_tcp_joint");
    candidates.push_back(arm_id_ + "_hand");
    candidates.push_back(arm_id_ + "_link8");
    candidates.push_back(arm_id_ + "_link7");

    ee_frame_id_ = model_.frames.size();
    for (const auto& name : candidates) {
        const auto id = model_.getFrameId(name);
        if (id != model_.frames.size()) {
            ee_frame_ = name;
            ee_frame_id_ = id;
            break;
        }
    }

    if (ee_frame_id_ == model_.frames.size()) {
        RCLCPP_ERROR(get_node()->get_logger(),
                                 "End-effector frame not found. Tried: '%s', '%s', '%s', '%s', '%s'.",
                                 candidates[0].c_str(), candidates[1].c_str(), candidates[2].c_str(),
                                 candidates[3].c_str(), candidates[4].c_str());
        return false;
    }

    RCLCPP_INFO(get_node()->get_logger(), "Using end-effector frame '%s'", ee_frame_.c_str());

    for (int i = 0; i < num_joints_; ++i) {
        const std::string joint_name = arm_id_ + "_joint" + std::to_string(i + 1);
        const auto joint_id = model_.getJointId(joint_name);
        if (joint_id == 0) {
            RCLCPP_ERROR(get_node()->get_logger(), "Joint '%s' not found in URDF.", joint_name.c_str());
            return false;
        }
        joint_ids_[i] = joint_id;
        joint_q_indices_[i] = model_.joints[joint_id].idx_q();
        joint_v_indices_[i] = model_.joints[joint_id].idx_v();
    }

    return true;
}

void GazeboCartesianImpedanceController::compute_current_pose_and_jacobian(
        Eigen::Quaterniond& current_orientation,
        Eigen::Vector3d& current_position,
        Eigen::Matrix<double, 6, 7>& jacobian) {
    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model_.nq);
    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(model_.nv);

    for (int i = 0; i < num_joints_; ++i) {
        q_full(joint_q_indices_[i]) = q_(i);
        v_full(joint_v_indices_[i]) = dq_filtered_(i);
    }

    pinocchio::forwardKinematics(model_, data_, q_full, v_full);
    pinocchio::updateFramePlacements(model_, data_);

    const pinocchio::SE3& oMe = data_.oMf[ee_frame_id_];
    current_position = oMe.translation();
    current_orientation = Eigen::Quaterniond(oMe.rotation());
    current_orientation.normalize();

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_full(6, model_.nv);
    J_full.setZero();
    pinocchio::computeFrameJacobian(
            model_, data_, q_full, ee_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_full);

    jacobian.setZero();
    for (int i = 0; i < num_joints_; ++i) {
        jacobian.col(i) = J_full.col(joint_v_indices_[i]);
    }
}

GazeboCartesianImpedanceController::Vector7d GazeboCartesianImpedanceController::compute_coriolis_and_gravity_torque() {
    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model_.nq);
    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(model_.nv);
    for (int i = 0; i < num_joints_; ++i) {
        q_full(joint_q_indices_[i]) = q_(i);
        v_full(joint_v_indices_[i]) = dq_filtered_(i);
    }

    Vector7d tau{Vector7d::Zero()};

    if (use_gravity_compensation_) {
        const Eigen::VectorXd tau_nle = pinocchio::nonLinearEffects(model_, data_, q_full, v_full);
        for (int i = 0; i < num_joints_; ++i) {
            tau(i) = tau_nle(joint_v_indices_[i]);
        }
    } else {
        pinocchio::computeCoriolisMatrix(model_, data_, q_full, v_full);
        const Eigen::VectorXd tau_c = data_.C * v_full;
        for (int i = 0; i < num_joints_; ++i) {
            tau(i) = tau_c(joint_v_indices_[i]);
        }
    }

    return tau;
}

controller_interface::InterfaceConfiguration
GazeboCartesianImpedanceController::command_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (int i = 1; i <= num_joints_; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
    }
    return config;
}

controller_interface::InterfaceConfiguration
GazeboCartesianImpedanceController::state_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (int i = 1; i <= num_joints_; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    }
    for (int i = 1; i <= num_joints_; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    }
    return config;
}

void GazeboCartesianImpedanceController::updateJointStates() {
    for (int i = 0; i < num_joints_; ++i) {
        const auto& position_interface = state_interfaces_.at(i);
        const auto& velocity_interface = state_interfaces_.at(num_joints_ + i);
        q_(i) = position_interface.get_value();
        dq_(i) = velocity_interface.get_value();
    }
}

void GazeboCartesianImpedanceController::deltaPoseCallback(
        const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(delta_pose_mutex_);

    delta_position_ = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    delta_orientation_ = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                                                                 msg->pose.orientation.y, msg->pose.orientation.z);

    new_delta_received_ = true;
}

Eigen::Vector3d GazeboCartesianImpedanceController::computeOrientationError(
        const Eigen::Quaterniond& orientation_d,
        const Eigen::Quaterniond& orientation) {
    Eigen::Quaterniond orientation_corrected = orientation;
    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation_corrected.coeffs() << -orientation.coeffs();
    }

    Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d);

    Eigen::Vector3d error;
    error << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

    return orientation_corrected.toRotationMatrix() * error;
}

controller_interface::return_type GazeboCartesianImpedanceController::update(
        const rclcpp::Time& /*time*/,
        const rclcpp::Duration& /*period*/) {
    updateJointStates();

    if (!pinocchio_ready_) {
        RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                                                    "Pinocchio model not ready; cannot run controller.");
        return controller_interface::return_type::ERROR;
    }

    const double kAlpha = 0.99;
    dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

    Eigen::Quaterniond current_orientation;
    Eigen::Vector3d current_position;
    Eigen::Matrix<double, 6, 7> jacobian;
    compute_current_pose_and_jacobian(current_orientation, current_position, jacobian);

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

    Eigen::Vector3d position_d;
    Eigen::Quaterniond orientation_d;
    {
        std::lock_guard<std::mutex> lock(delta_pose_mutex_);
        if (new_delta_received_) {
            Eigen::Vector3d target_position = current_position + delta_position_;
            const double max_position_error = 0.12;
            Eigen::Vector3d error_vec = target_position - current_position;
            const double error_norm = error_vec.norm();
            if (error_norm > max_position_error) {
                target_position = current_position + error_vec * (max_position_error / error_norm);
            }

            const double alpha = 0.3;
            desired_position_ = desired_position_ * (1.0 - alpha) + target_position * alpha;

            desired_orientation_ = delta_orientation_ * current_orientation;
            desired_orientation_.normalize();
            new_delta_received_ = false;
        }
        position_d = desired_position_;
        orientation_d = desired_orientation_;
    }

    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << current_position - position_d;
    error.tail(3).setZero();

    static int counter = 0;
    if (++counter % 100 == 0) {
        RCLCPP_INFO(get_node()->get_logger(),
                                "Position error: [%.4f, %.4f, %.4f], Orientation error: [%.4f, %.4f, %.4f]",
                                error(0), error(1), error(2), error(3), error(4), error(5));
    }

    Eigen::Matrix<double, 6, 1> velocity = jacobian * dq_filtered_;
    const Vector7d coriolis_and_gravity = compute_coriolis_and_gravity_torque();

    Eigen::Matrix<double, 6, 1> force = -k_gains_.cwiseProduct(error) - d_gains_.cwiseProduct(velocity);
    Vector7d tau_d = jacobian.transpose() * force + coriolis_and_gravity;

    const double max_torque_rate = 50.0;
    const double delta_tau_max = max_torque_rate * 0.001;

    for (int i = 0; i < num_joints_; ++i) {
        double delta_tau = tau_d(i) - tau_commanded_(i);
        delta_tau = std::clamp(delta_tau, -delta_tau_max, delta_tau_max);
        tau_commanded_(i) += delta_tau;
    }

    for (int i = 0; i < num_joints_; ++i) {
        command_interfaces_[i].set_value(tau_commanded_(i));
    }

    return controller_interface::return_type::OK;
}

CallbackReturn GazeboCartesianImpedanceController::on_init() {
    try {
        auto_declare<std::string>("arm_id", "fr3");
        auto_declare<std::vector<double>>("k_gains", {50.0, 50.0, 50.0, 5.0, 5.0, 5.0});
        auto_declare<std::vector<double>>("d_gains", {3.0, 3.0, 3.0, 2.0, 2.0, 2.0});
        auto_declare<bool>("load_gripper", true);
        auto_declare<std::string>("ee_frame", "");
        auto_declare<bool>("use_gravity_compensation", true);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Exception during on_init: %s", e.what());
        return CallbackReturn::ERROR;
    }

    return CallbackReturn::SUCCESS;
}

bool GazeboCartesianImpedanceController::assign_parameters() {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    is_gripper_loaded_ = get_node()->get_parameter("load_gripper").as_bool();

    auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
    auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

    if (k_gains.empty()) {
        RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
        return false;
    }
    if (k_gains.size() != 6) {
        RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size 6 but is of size %ld", k_gains.size());
        return false;
    }
    if (d_gains.empty()) {
        RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
        return false;
    }
    if (d_gains.size() != 6) {
        RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size 6 but is of size %ld", d_gains.size());
        return false;
    }

    for (int i = 0; i < 6; ++i) {
        k_gains_(i) = k_gains.at(i);
        d_gains_(i) = d_gains.at(i);
    }

    RCLCPP_INFO(get_node()->get_logger(), "K gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
                            k_gains_(0), k_gains_(1), k_gains_(2), k_gains_(3), k_gains_(4), k_gains_(5));
    RCLCPP_INFO(get_node()->get_logger(), "D gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
                            d_gains_(0), d_gains_(1), d_gains_(2), d_gains_(3), d_gains_(4), d_gains_(5));

    return true;
}

CallbackReturn GazeboCartesianImpedanceController::on_configure(
        const rclcpp_lifecycle::State& /*previous_state*/) {
    if (!assign_parameters()) {
        return CallbackReturn::FAILURE;
    }

    {
        auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
                "service_server/set_full_collision_behavior");
        auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();

        if (!client->wait_for_service(robot_utils::time_out)) {
            RCLCPP_WARN(get_node()->get_logger(),
                                    "Collision behavior service not available; skipping (expected in Gazebo). ");
        } else {
            auto future_result = client->async_send_request(request);
            if (future_result.wait_for(robot_utils::time_out) != std::future_status::ready) {
                RCLCPP_WARN(get_node()->get_logger(), "Timed out setting collision behavior; continuing.");
            } else {
                auto response = future_result.get();
                if (!response) {
                    RCLCPP_WARN(get_node()->get_logger(),
                                            "Collision behavior service returned null response; continuing.");
                } else {
                    RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior request sent.");
                }
            }
        }
    }

    auto parameters_client =
            std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
    if (!parameters_client->wait_for_service(robot_utils::time_out)) {
        RCLCPP_FATAL(get_node()->get_logger(),
                                 "robot_state_publisher parameter service not available.");
        return CallbackReturn::ERROR;
    }

    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
        robot_description_ = result[0].value_to_string();
    } else {
        RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
    }

    arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

    if (!build_pinocchio_model_from_urdf(robot_description_)) {
        return CallbackReturn::ERROR;
    }
    if (!initialize_pinocchio_joint_and_frame_mapping()) {
        return CallbackReturn::ERROR;
    }
    pinocchio_ready_ = true;

    delta_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/delta_pose", rclcpp::QoS(1).best_effort(),
            std::bind(&GazeboCartesianImpedanceController::deltaPoseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_node()->get_logger(),
                            "Gazebo Cartesian Impedance Controller configured. Subscribing to /delta_pose");

    return CallbackReturn::SUCCESS;
}

CallbackReturn GazeboCartesianImpedanceController::on_activate(
        const rclcpp_lifecycle::State& /*previous_state*/) {
    initialization_flag_ = true;
    dq_filtered_.setZero();

    RCLCPP_INFO(get_node()->get_logger(), "Gazebo Cartesian Impedance Controller activated.");

    return CallbackReturn::SUCCESS;
}

CallbackReturn GazeboCartesianImpedanceController::on_deactivate(
        const rclcpp_lifecycle::State& /*previous_state*/) {
    return CallbackReturn::SUCCESS;
}

}  // namespace franka_iri_controllers

PLUGINLIB_EXPORT_CLASS(franka_iri_controllers::GazeboCartesianImpedanceController,
                                             controller_interface::ControllerInterface)
