/*
File Name: slamBase.cpp
Author: xiang gao
*/
#include "slamBase.h"

PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth,
								 CAMERA_INTRINSIC_PARAMETERS& camera)
{
	PointCloud::Ptr cloud(new PointCloud);

	for (int m=0;m<depth.rows;m++)
		for (int n=0;n<depth.cols;n++)
		{
			//get the value at (m,n)
			ushort d=depth.ptr<ushort>(m)[n];
			//maybe d has no value, if so jump over this point
			if (d==0)
				continue;
			//if d has value, add one point to PC
			PointT p;

			//calculate the space vector of this point
			p.z=double (d)/camera.scale;
			p.x=(n-camera.cx)*p.z/camera.fx;
			p.y=(m-camera.cy)*p.z/camera.fy;

			//get color from rgb.png
			//for the rgb pic has 3ch, use the following order
			p.b=rgb.ptr<uchar>(m)[n*3];
			p.g=rgb.ptr<uchar>(m)[n*3+1];
			p.r=rgb.ptr<uchar>(m)[n*3+2];

			//add p to the PC
			cloud->points.push_back(p);
		}
	//set and save PC
	cloud->height=1;
	cloud->width=cloud->points.size();
	cloud->is_dense=false;

	return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	cv::Point3f p;//3D point
	p.z=double(point.z)/camera.scale;
	p.x=(point.x-camera.cx)*p.z/camera.fx;
	p.y=(point.y-camera.cy)*p.z/camera.fy;
	return p;
}

//computeKeyPointsAndDesp
void computeKeyPointsAndDesp (FRAME& frame, string detector, string descriptor)
{
	cv::Ptr<cv::FeatureDetector> _detector;
	cv::Ptr<cv::DescriptorExtractor> _descriptor;

	cv::initModule_nonfree();
	_detector=cv::FeatureDetector::create(detector.c_str());
	_descriptor=cv::DescriptorExtractor::create(descriptor.c_str());

	if (!_detector||!_descriptor)
	{
		cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
		return;
	}

	_detector->detect(frame.rgb,frame.kp);
	_descriptor->compute(frame.rgb,frame.kp,frame.desp);

	return;
}

//estimateMotion calculate the motion between two frames
//input:frame1 & frame2
//output:rvec & tvec
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	static ParameterReader pd;
	vector<cv::DMatch> matches;
	//cv::FlannBasedMatcher matcher;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(frame1.desp,frame2.desp,matches);

	cout<<"find total "<<matches.size()<<" matches."<<endl;
	vector<cv::DMatch> goodMatches;
	double minDis=9999;
	double good_match_threshold=atof(pd.getData("good_match_threshold").c_str());
	for (size_t i=0;i<matches.size();i++)
	{
		if (matches[i].distance<minDis)
			minDis=matches[i].distance;
	}
	for (size_t i=0;i<matches.size();i++)
	{
		if (matches[i].distance<good_match_threshold*minDis)
			goodMatches.push_back(matches[i]);
	}

	cout<<"good matches: "<<goodMatches.size()<<endl;
	//the 3D point of the frame1
	vector<cv::Point3f> pts_obj;
	//the pic point of the frame2
	vector<cv::Point2f> pts_img;

	//Camera Intrinsic Parameters
	for (size_t i=0;i<goodMatches.size();i++)
	{
		//query for 1st, train for 2nd
		cv::Point2f p=frame1.kp[goodMatches[i].queryIdx].pt;
		//here be careful! x is to the right, y points down=>so y is row and x is column
		ushort d=frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
		if (d==0)//if no depth data, jump over this GM, be careful to avoid bug!
			continue;
		pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));

		//transform (u,v,d) to (x,y,z)
		cv::Point3f pt(p.x,p.y,d);
		cv::Point3f pd=point2dTo3d(pt,camera);
		pts_obj.push_back(pd);
	}

	double camera_matrix_data[3][3]={
		{camera.fx,0,camera.cx},
		{0,camera.fy,camera.cy},
		{0,0,1}
	};

	cout<<"solving pnp"<<endl;
	//build camera matrix
	cv::Mat cameraMatrix(3,3,CV_64F,camera_matrix_data);
	cv::Mat rvec,tvec,inliers;
	//solve pnp
	if (pts_obj.size()!=0&&pts_img.size()!=0)
		cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec,false,
			100,1.0,100,inliers);
	else
	{
		rvec=cv::Mat::zeros(3,3,CV_64F);
		tvec=cv::Mat::zeros(3,1,CV_64F);
		inliers=cv::Mat::zeros(1,1,CV_32S);
	}

	RESULT_OF_PNP result;
	result.rvec=rvec;
	result.tvec=tvec;
	result.inliers=inliers.rows;//here inliers don't accord to the right order in GM

	return result;
}

//cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Mat R;
	cv::Rodrigues(rvec,R);
	Eigen::Matrix3d r;
	cv::cv2eigen(R,r);

	//transform tvec&rvec to transformation
	Eigen::Isometry3d T=Eigen::Isometry3d::Identity();

	Eigen::AngleAxisd angle(r);
	Eigen::Translation<double,3> trans(tvec.at<double>(0),tvec.at<double>(1),
		tvec.at<double>(2,0));
	T=angle;
	T(0,3)=tvec.at<double>(0);
	T(1,3)=tvec.at<double>(1);
	T(2,3)=tvec.at<double>(2);
	
	return T;
}

//joinPointCloud 
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame,
							   Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	PointCloud::Ptr newCloud=image2PointCloud(newFrame.rgb,newFrame.depth,camera);

	//concatenate PC
	PointCloud::Ptr output(new PointCloud());
	pcl::transformPointCloud(*original,*output,T.matrix());
	*newCloud+=*output;

	//Voxel grid: use its filter to reduce the sampling rate
	static pcl::VoxelGrid<PointT> voxel;
	static ParameterReader pd;
	double gridsize=atof(pd.getData("voxel_grid").c_str());
	voxel.setLeafSize(gridsize,gridsize,gridsize);
	voxel.setInputCloud(newCloud);
	PointCloud::Ptr tmp(new PointCloud());
	voxel.filter(*tmp);

	return tmp;
}
