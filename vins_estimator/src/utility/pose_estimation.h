#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <stdexcept>
#include <limits>

namespace pose {

struct PoseResult {
    Eigen::Matrix3d R_cw;   // 世界 -> 相机
    Eigen::Vector3d t_cw;
    Eigen::Matrix3d R_wc;   // 相机 -> 世界 (相机位姿)
    Eigen::Vector3d t_wc;   // 相机光心在世界坐标系下的位置
    double scale = 1.0;     // 估计的尺度（可选）
};

// 3D-3D 位姿估计 (Umeyama / SVD)
// P_world: Nx3 世界坐标
// M_cam:   Nx3 相机坐标
// with_scale: 是否估计尺度
PoseResult estimatePose(const Eigen::MatrixXd& P_world,
                        const Eigen::MatrixXd& M_cam,
                        bool with_scale = false);

// 带 RANSAC 的鲁棒版本
PoseResult estimatePoseRANSAC(const Eigen::MatrixXd& P_world,
                              const Eigen::MatrixXd& M_cam,
                              double thresh = 0.01,
                              int iters = 1000,
                              int seed = 0,
                              std::vector<int>* inliers_out = nullptr);

} // namespace pose