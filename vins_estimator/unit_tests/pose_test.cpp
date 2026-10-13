#include "../src/utility/pose_estimation.h"

#include <iostream>
#include <random>

int main() {
    // ---- 构造真值 ----
    Eigen::Matrix3d R_true;   // 世界 -> 相机
    R_true = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())
           * Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY())
           * Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX());
    Eigen::Vector3d t_true(0.5, -0.3, 1.2);

    // ---- 生成 N 个世界坐标点 ----
    const int N = 20;
    Eigen::MatrixXd P_world(N, 3);
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        P_world(i, 0) = nd(rng);
        P_world(i, 1) = nd(rng);
        P_world(i, 2) = nd(rng);
    }

    // ---- 生成相机坐标 (加噪声) ----
    Eigen::MatrixXd M_cam(N, 3);
    std::normal_distribution<double> noise(0.0, 0.001);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p = P_world.row(i).transpose();
        Eigen::Vector3d m = R_true * p + t_true;
        m += Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
        M_cam.row(i) = m.transpose();
    }

    // ---- 估计位姿 ----
    pose::PoseResult res = pose::estimatePose(P_world, M_cam);

    std::cout << "=== Estimated world->camera ===\n";
    std::cout << "R_cw:\n" << res.R_cw << "\n";
    std::cout << "t_cw: " << res.t_cw.transpose() << "\n\n";

    std::cout << "=== Camera pose in world ===\n";
    std::cout << "R_wc:\n" << res.R_wc << "\n";
    std::cout << "t_wc (camera center in world): "
              << res.t_wc.transpose() << "\n\n";

    std::cout << "=== Ground truth ===\n";
    std::cout << "R_true:\n" << R_true << "\n";
    std::cout << "t_true: " << t_true.transpose() << "\n";

    // ---- 验证 ----
    std::cout << "\n=== Verification ===\n";
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p = P_world.row(i).transpose();
        Eigen::Vector3d m_est = res.R_cw * p + res.t_cw;
        double err = (m_est - M_cam.row(i).transpose()).norm();
        max_err = std::max(max_err, err);
    }
    std::cout << "Max reprojection error: " << max_err << "\n";

    return 0;
}