#include "pose_estimation.h"

#include <iostream>

namespace pose {

PoseResult estimatePose(const Eigen::MatrixXd& P_world,
                        const Eigen::MatrixXd& M_cam,
                        bool with_scale)
{
    if (P_world.rows() != M_cam.rows() || P_world.cols() != 3 || M_cam.cols() != 3) {
        throw std::invalid_argument("P_world and M_cam must be Nx3 matrices with same N");
    }
    const int N = static_cast<int>(P_world.rows());
    if (N < 3) {
        throw std::invalid_argument("At least 3 points are required");
    }

    // 1. 质心
    Eigen::Vector3d c_P = P_world.colwise().mean().transpose();
    Eigen::Vector3d c_M = M_cam.colwise().mean().transpose();

    // 2. 去质心
    Eigen::MatrixXd P0 = P_world.rowwise() - c_P.transpose();
    Eigen::MatrixXd M0 = M_cam.rowwise()   - c_M.transpose();

    // 3. 协方差矩阵 H = sum(P0_i * M0_i^T)  = P0^T * M0
    Eigen::Matrix3d H = P0.transpose() * M0;

    // 4. SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d S = svd.singularValues();

    // 5. 处理反射，保证 det(R) = +1
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((V * U.transpose()).determinant() < 0) {
        D(2, 2) = -1.0;
    }

    // 6. 旋转矩阵
    Eigen::Matrix3d R_cw = V * D * U.transpose();

    // 7. 尺度（可选）
    double scale = 1.0;
    if (with_scale) {
        double var_P = P0.array().square().sum() / static_cast<double>(N);
        double trace_S = (S.array() * D.diagonal().array()).sum();
        if (var_P > std::numeric_limits<double>::epsilon()) {
            scale = trace_S / var_P;
        }
    }

    // 8. 平移
    Eigen::Vector3d t_cw = c_M - scale * R_cw * c_P;

    // 9. 相机在世界坐标系下的位姿
    PoseResult res;
    res.scale = scale;
    res.R_cw  = R_cw;
    res.t_cw  = t_cw;
    res.R_wc  = R_cw.transpose();
    res.t_wc  = -R_cw.transpose() * t_cw / scale;  // 若 scale != 1 时需注意
    return res;
}

PoseResult estimatePoseRANSAC(const Eigen::MatrixXd& P_world,
                              const Eigen::MatrixXd& M_cam,
                              double thresh,
                              int iters,
                              int seed,
                              std::vector<int>* inliers_out)
{
    const int N = static_cast<int>(P_world.rows());
    if (N < 3) throw std::invalid_argument("At least 3 points are required");

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> best_inliers;
    PoseResult best_pose;

    for (int it = 0; it < iters; ++it) {
        // 随机选 3 个点
        int i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Eigen::MatrixXd P3(3, 3), M3(3, 3);
        P3 << P_world.row(i0), P_world.row(i1), P_world.row(i2);
        M3 << M_cam.row(i0),   M_cam.row(i1),   M_cam.row(i2);

        PoseResult pose;
        try {
            pose = estimatePose(P3, M3, false);
        } catch (...) {
            continue;
        }

        // 计算残差
        Eigen::MatrixXd M_pred = (pose.R_cw * P_world.transpose()).transpose()
                                 .rowwise() + pose.t_cw.transpose();
        std::vector<int> inliers;
        for (int i = 0; i < N; ++i) {
            double err = (M_pred.row(i) - M_cam.row(i)).norm();
            if (err < thresh) inliers.push_back(i);
        }

        if (inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
            best_pose = pose;
        }
    }

    if (best_inliers.size() < 3) {
        throw std::runtime_error("RANSAC failed: not enough inliers");
    }

    // 用所有内点重新精化
    Eigen::MatrixXd P_in(best_inliers.size(), 3);
    Eigen::MatrixXd M_in(best_inliers.size(), 3);
    for (size_t k = 0; k < best_inliers.size(); ++k) {
        P_in.row(k) = P_world.row(best_inliers[k]);
        M_in.row(k) = M_cam.row(best_inliers[k]);
    }

    PoseResult refined = estimatePose(P_in, M_in, false);

    if (inliers_out) *inliers_out = best_inliers;
    return refined;
}

} // namespace pose