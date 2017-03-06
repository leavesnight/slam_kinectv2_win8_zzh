/*
File Name:RGBDSLAM/slamBase.h
Author:xiang gao
Explanation:base function (c type)
*/
#pragma once

//you have to use import when there exists a static variant in your class
//you have to use export when you use class
//and you can choose this way instead of def
#define DLL_EXPORT
#ifdef DLL_EXPORT
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

//header files
//c++ standard lib
#include <fstream>
#include <vector>
//eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//feature detection mode
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>//here for solvePnPRansac func
//cv2eigen
#include <opencv2/core/eigen.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//transformPC
#include <pcl/common/transforms.h>
//voxel grid
#include <pcl/filters/voxel_grid.h>

//typedef
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//intrinsic structure of the cammera
struct CAMERA_INTRINSIC_PARAMETERS
{
	double cx,cy,fx,fy,scale;
};

//Function Interface
//image2PointCloudtranform rgb pic to PC
PointCloud::Ptr image2PointCloud(cv::Mat& rgb,cv::Mat& depth,
								 CAMERA_INTRINSIC_PARAMETERS& camera);

//point2dTo3d find the 3d coordinate of a point from the pic
//input: 3d point Point3f(u,v,d); output: 3d point Point3f(x,y,z);
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

//4¡ª¡ªframe structure
struct FRAME
{
	cv::Mat rgb,depth;//the color and depth frame of the frame
	cv::Mat desp;//the descriptor of the feature
	vector<cv::KeyPoint> kp;//key points' array
	int frameID;
};

//PnP result
struct RESULT_OF_PNP
{
	cv::Mat rvec,tvec;
	int inliers;
};

//computeKeyPointsAndDesp get the KP & FD at the same time
void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor);

//estimateMotion calculate the motion between two frames
//input:frame1 & frame2, Camera Intrinsic parameters
DLL_EXPORT RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

class ParameterReader
{
public:
	map<string,string> data;//the UDT_MAP_STRING_STRING class data
	ParameterReader(string filename="./parameters.txt")
	{
		ifstream fin(filename.c_str());
		if (!fin)
		{
			cerr<<"parameter file does not exist."<<endl;
			return;
		}
		while (!fin.eof())//when getline cannot get anything, eof will become T after one continue
		{
			string str;
			getline(fin,str);
			if (str[0]=='#')
			{
				//'#' is annotation
				continue;
			}

			int pos=str.find("=");
			if (pos==-1)
				continue;
			string key=str.substr(0,pos);
			string value=str.substr(pos+1,str.length());
			data[key]=value;

			if(!fin.good())//if the file is not well opened
				break;
		}
	}
	string getData(string key)
	{
		map<string,string>::iterator iter=data.find(key);
		if (iter==data.end())
		{
			cerr<<"Parameter name "<<key<<" not found!"<<endl;
			return string("NOT_FOUND");
		}
		return iter->second;
	}
};

//5-cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);
//joinPointCloud
//input:original PC, newFrame & Transformation with camera matrix
//output:new PC adding the newFrame
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame,
					Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera);
