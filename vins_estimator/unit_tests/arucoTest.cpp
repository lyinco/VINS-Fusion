// arucoTest.cpp
#include <cassert>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../src/utility/aruco_utility.h"


int main() {
    // 4.2 里字典创建方式
    cv::Ptr<cv::aruco::Dictionary> dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);

    // 用 drawMarker 生成 marker 图（无白边、无背景）
    cv::Mat marker_img;
    cv::aruco::drawMarker(dict, 7, 200, marker_img, 1);

    // 手动贴到白色背景上，留出静默区（quiet zone）
    cv::Mat canvas = cv::Mat::ones(400, 400, CV_8UC1) * 255;
    marker_img.copyTo(canvas(cv::Rect(100, 100, 200, 200)));

    // 测试正常检测
    auto result = detectArucoCorners(canvas, dict);
    printf("Detected %zu markers\n", result.size());
    printf("Marker ID 7 has corners: (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)\n",
           result[7][0].x, result[7][0].y, result[7][1].x, result[7][1].y,
           result[7][2].x, result[7][2].y, result[7][3].x, result[7][3].y);
    assert(result.size() == 1);
    assert(result.count(7) == 1);
    assert(result[7].size() == 4);

    // 将检测角点画到图片上
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    for (const auto& kv : result) {
        ids.push_back(kv.first);
        corners.push_back(kv.second);
    }
    cv::Mat vis;
    cv::cvtColor(canvas, vis, cv::COLOR_GRAY2BGR);
    // 5. 画四个角点
    for (const auto& kv : result) {
        int id = kv.first;
        const std::vector<cv::Point2f>& corners = kv.second; // 4 个角点

        // 用不同颜色区分 4 个角点，方便验证顺序
        // 顺序：左上、右上、右下、左下
        cv::circle(vis, corners[0], 5, cv::Scalar(0, 0, 255),   -1); // 红：左上
        cv::circle(vis, corners[1], 5, cv::Scalar(0, 255, 0),   -1); // 绿：右上
        cv::circle(vis, corners[2], 5, cv::Scalar(255, 0, 0),   -1); // 蓝：右下
        cv::circle(vis, corners[3], 5, cv::Scalar(0, 255, 255), -1); // 黄：左下
        // 在第一个角点旁写上 marker ID
        cv::putText(vis, std::to_string(id),
                    corners[0] + cv::Point2f(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("Aruco Marker Test", vis);
    cv::waitKey(0);

    // 测试空图
    auto empty_result = detectArucoCorners(cv::Mat(), dict);
    assert(empty_result.empty());


    

    std::cout << "All tests passed!" << std::endl;
    return 0;
}