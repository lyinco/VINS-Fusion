// aruco_utility.h
#ifndef ARUCO_UTILITY_H
#define ARUCO_UTILITY_H

#include <map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>   // OpenCV 4.2 的 ArUco 头文件

std::map<int, std::vector<cv::Point2f>> detectArucoCorners(
        const cv::Mat& img,
        const cv::Ptr<cv::aruco::Dictionary>& dict);

#endif // ARUCO_UTILITY_H