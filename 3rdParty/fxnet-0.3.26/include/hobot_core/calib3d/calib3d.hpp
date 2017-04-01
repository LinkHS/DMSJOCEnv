/*
 * calib3d.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_CALIB3D_HPP_
#define FXNET_CALIB3D_HPP_

//#define  USE_OPENCV

#ifdef USE_OPENCV
#include <vector>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "hobot_core/base/base_common.hpp"

namespace hbot {
namespace calib3d {

class PNPSolver{
public:
	PNPSolver():img_h_(0),img_w_(0){
		rvec_ = cv::Mat(1,3,CV_64FC1);
		tvec_ = cv::Mat(1,3,CV_64FC1);
		distCoeff_ = cv::Mat(1,4,CV_64FC1);
		rotM_ = cv::Mat(3,3,CV_64FC1);
		pM_ = cv::Mat(3,4,CV_64FC1);
	};
	~PNPSolver(){};

	cv::Vec3f Solve(cv::Mat& img, cv::Mat& modelPoints, cv::Mat& img_points);
	void ShowReprojectedPoints(cv::Mat& img,cv::Mat& modelPoint);
private:
	int img_h_, img_w_;
	cv::Mat camMatrix_;
	cv::Mat rvec_ ,tvec_ ;
	cv::Mat distCoeff_;
	cv::Mat rotM_;
	cv::Mat pM_;
};



}  // namespace calib3d
}  // namespace hbot
#endif

#endif /* FXNET_CALIB3D_HPP_ */
