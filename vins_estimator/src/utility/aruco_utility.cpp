// aruco_utility.cpp
#include "aruco_utility.h"

std::map<int, std::vector<cv::Point2f>> detectArucoCorners(
        const cv::Mat& img,
        const cv::Ptr<cv::aruco::Dictionary>& dict)
{
    std::map<int, std::vector<cv::Point2f>> arucoCorners;

    if (img.empty() || !dict) {
        return arucoCorners;
    }

    // 4.2 里 DetectorParameters 用 create() 创建
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::aruco::DetectorParameters::create();

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;

    // 4.2 的函数签名：detectMarkers(image, dictionary, corners, ids, params, rejected)
    cv::aruco::detectMarkers(img, dict, corners, ids, params, rejected);

    for (size_t i = 0; i < ids.size(); ++i) {
        arucoCorners[ids[i]] = corners[i];
    }

    return arucoCorners;
}


