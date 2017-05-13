/*
File Name:slam_KV2.cpp
Author:zzh
*/
#define insert_in(a,b,c) ((a)<(b)?(b):(a)>(c)?(c):(a))

//Kinect lib
#include <Kinect.h>

//C++ standard library
#include <iostream>
//#include <fstream>//it's included in slamBase.h
#include <sstream>//stream for string
#include <fstream>//stream for pColorSPd.txt
#include <Windows.h>//change the console output color
using namespace std;

//SLAM based head files
#include "slamBase.h"

//OpenCV additional HF
#include <opencv2/opencv.hpp>//to use IplImage class

//PCL additional HF
#include <pcl/visualization/cloud_viewer.h>//show the PC
#include <pcl/filters/passthrough.h>//use the z direction filed filter

//g2o HF
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

//get default camera parameters
CAMERA_INTRINSIC_PARAMETERS getDefaultCamera();
//Read one frame with given index
FRAME readFrame(int index, ParameterReader& pd);
FRAME readFrame(int index, ParameterReader& pd,IColorFrameReader* colorfr,IDepthFrameReader* depthfr,
				cv::Size sizeC,cv::Size sizeD,ColorSpacePoint* pColorSP,bool bFirst=false);
//measure the range of motion
double normofTransform(cv::Mat rvec,cv::Mat tvec);

//the result defination of checking two frames
enum CHECK_RESULT{NOT_MATCHED=0,TOO_FAR_AWAY,TOO_CLOSE,KEYFRAME};
//function declaration
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false);
//check nearby loops
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);
//check random loops
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

//change the console output color
inline ostream& red(std::ostream& s)//create this function is the same as reload ostream1 << ostream2
{
	HANDLE hStdout=::GetStdHandle(STD_OUTPUT_HANDLE);
	::SetConsoleTextAttribute(hStdout,FOREGROUND_RED|FOREGROUND_INTENSITY);
	return s;
}
inline ostream& green(std::ostream& s)
{
	HANDLE hStdout=::GetStdHandle(STD_OUTPUT_HANDLE);
	::SetConsoleTextAttribute(hStdout,FOREGROUND_GREEN|FOREGROUND_INTENSITY);
	return s;
}
inline ostream& white(std::ostream& s)
{
	HANDLE hStdout=::GetStdHandle(STD_OUTPUT_HANDLE);
	::SetConsoleTextAttribute(hStdout,FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_BLUE);
	return s;
}

ICoordinateMapper* pMapper;
//when mouse clicked, show the coordinate
//IplImage* src=nullptr;
cv::Mat srcMat;
void on_mouse(int event,int x,int y,int flags,void* ustc){
	//CvFont font;
	//cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1,1,0,2,CV_AA);
	if (event==CV_EVENT_LBUTTONDOWN){
		//CvPoint pt=cvPoint(x*4/5,y*540/695);//???
		CvPoint pt=cvPoint(x,y);
		char temp[16];
		sprintf(temp,"(%d,%d)",x,y);
		//src=&IplImage(srcMat);
		//cvPutText(src,temp,pt,&font,cvScalar(255,255,255,0));
		cv::putText(srcMat,temp,pt,CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255,0),2,CV_AA);
		//cvCircle(src,pt,4,cvScalar(255,0,0,0),CV_FILLED,CV_AA);
		cv::circle(srcMat,pt,4,cv::Scalar(255,0,0,0),CV_FILLED,CV_AA);
		//cvShowImage("Test Color",src);
		cv::imshow("Test Color",srcMat);
		cv::waitKey(500);
		//printf("%d,%d, Color\n",x,y);
		cout<<green<<x<<","<<y<<" Color"<<white<<endl;
	}
}
void on_mouse2(int event,int x,int y,int flags,void* ustc){
	if (event==CV_EVENT_LBUTTONDOWN){
		//printf("%d,%d, Depth\n",x,y);
		cout<<green<<x<<","<<y<<" Depth"<<white<<endl;
	}
}

vector<int> loopIndexPairs;
int g_count=0;

//main function
int main(int argc, char** argv){
	//Open KV2
	//read color and depth data from kinect and tranform them to PC
	cout<<"Hello SLAM!"<<endl;

	IKinectSensor* pKinect;//apply for a pointer to the sensor class
	HRESULT hr=::GetDefaultKinectSensor(&pKinect);//get a default sensor
	if (FAILED(hr)){
		printf("No Kinect connect to your pc!\n");
		return 0;//goto ENDSTOP;
	}
	BOOLEAN bIsOpen=0;
	pKinect->get_IsOpen(&bIsOpen);//find if kinect is opened
	printf("bIsOpen: %d\n",bIsOpen);

	if (!bIsOpen){//if kinect isn't opened, try to open it
		hr=pKinect->Open();
		if (FAILED(hr)){
			printf("Kinect Open Failed!\n");
			return 0;
		}
		printf("Kinect is opened!But it needs sometime to work!\n");
		printf("Wait for 3000ms...\n");
		Sleep(3000);
	}
	bIsOpen=0;
	pKinect->get_IsOpen(&bIsOpen);//find if kinect is opened again
	printf("bIsOpen: %d\n",bIsOpen);
	BOOLEAN bAvailable=0;
	pKinect->get_IsAvailable(&bAvailable);//check if kinect is ready to use
	printf("bAvailable: %d\n",bAvailable);

	//Print the image size
	IColorFrameSource* colorfs=nullptr;
	pKinect->get_ColorFrameSource(&colorfs);//get the color source
	IColorFrameReader* colorfr=nullptr;
	colorfs->OpenReader(&colorfr);//bind the reader to the source
	IFrameDescription* frameds=nullptr;
	colorfs->get_FrameDescription(&frameds);//get the information of the color image
	int height,width;
	frameds->get_Height(&height);
	frameds->get_Width(&width);
	printf("Frame: %d %d\n",height,width);

	IDepthFrameSource* depthfs=NULL;
	pKinect->get_DepthFrameSource(&depthfs);
	IDepthFrameReader* depthfr=NULL;
	depthfs->OpenReader(&depthfr);
	IFrameDescription* depthfd=NULL;
	depthfs->get_FrameDescription(&depthfd);
	int hei2,wid2;
	depthfd->get_Height(&hei2);
	depthfd->get_Width(&wid2);
	printf("Depth: %d %d\n",hei2,wid2);

	//get pColorSP for KV2,actually it's pDepthSP
	int count=height*width;
	ColorSpacePoint* pColorSP=new ColorSpacePoint[count];
//	ICoordinateMapper* pMapper;
	pKinect->get_CoordinateMapper(&pMapper);
	CameraIntrinsics camIntr;
	pMapper->GetDepthCameraIntrinsics(&camIntr);
	cout<<camIntr.FocalLengthX<<" "<<camIntr.FocalLengthY<<" "
		<<camIntr.PrincipalPointX<<" "<<camIntr.PrincipalPointY<<" "
		<<camIntr.RadialDistortionSecondOrder<<" "<<camIntr.RadialDistortionFourthOrder
		<<" "<<camIntr.RadialDistortionSixthOrder<<endl;

	//SLAM
	ParameterReader pd;//get the basic slam information
	vector<FRAME> keyframes;//keyframes array
	//save SLAM images from kinectv2
	int currIndex=0;
	cout<<"Do you want to take photos?Y or N"<<endl;
	string strAns;
	cin>>strAns;
	if (strAns=="Y"||strAns=="y"){
		//Create windows
		//API:http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#cv2.imread
		cv::namedWindow("Test Color");//call a window
		cv::namedWindow("Test Depth");
		cvSetMouseCallback("Test Color",on_mouse);
		cvSetMouseCallback("Test Depth",on_mouse2);
		do
		{
			//Sleep(300);
			cout<<"Reading files"<<currIndex<<endl;
			FRAME f=readFrame(currIndex,pd,colorfr,depthfr,
			cv::Size(width,height),cv::Size(wid2,hei2),pColorSP,!currIndex);//read the KV2 to get currFrame
			if (f.frameID!=-1)
				currIndex++;
		}while (currIndex<400);
	}
	//initialize
	cout<<"Initializeing..."<<endl;
	int startIndex=atoi(pd.getData("start_index").c_str());
	int endIndex=atoi(pd.getData("end_index").c_str());
	currIndex=startIndex;
	FRAME currFrame=readFrame(currIndex,pd);
	string detector=pd.getData("detector");
	string descriptor=pd.getData("descriptor");	
	computeKeyPointsAndDesp(currFrame,detector,descriptor);

	//new initialization about g2o
	//initialize the solver
	SlamLinearSolver* linearSolver=new SlamLinearSolver();
	linearSolver->setBlockOrdering(false);
	SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(blockSolver);

	g2o::SparseOptimizer globalOptimizer;//final we use this variable
	globalOptimizer.setAlgorithm(solver);
	//no need for debug info
	globalOptimizer.setVerbose(false);

	//add the 1st vertex to the globalOptimizer
	g2o::VertexSE3* v=new g2o::VertexSE3();
	v->setId(currIndex);
	v->setEstimate(Eigen::Isometry3d::Identity());//estimate uses an indentity matrix
	v->setFixed(true);//the 1st vertex is fixed, no need to be optimized
	globalOptimizer.addVertex(v);

	keyframes.push_back(currFrame);//add f0 to F
	bool check_loop_closure=pd.getData("check_loop_closure")==string("yes");

	/*cv::Size shSize=cv::Size(width/2,height/2);//notice Size uses width first!
	cv::Mat rgbsh(shSize,CV_8UC3);
	cv::resize(currFrame.rgb,rgbsh,shSize);
	for (int i=0;i<rgbsh.rows;++i){
		for (int j=0;j<rgbsh.cols;++j){
			rgbsh.ptr<uchar>(i)[j*3]=rgbsh.ptr<uchar>(i)[j*3]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
			rgbsh.ptr<uchar>(i)[j*3+1]=rgbsh.ptr<uchar>(i)[j*3+1]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
			rgbsh.ptr<uchar>(i)[j*3+2]=rgbsh.ptr<uchar>(i)[j*3+2]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
		}
	}
	srcMat=rgbsh.clone();
	imshow("Test Color",rgbsh);
	cv::waitKey(1);
	cvSetMouseCallback("Test Color",on_mouse);
	if (currFrame.frameID==currIndex)
		system("pause");*/

CYCLE:
	clock_t nTmStart=clock(),g_count1=0,g_count2=0;
	//the id of last frame is already recorded in FRAME structure
	for (currIndex=startIndex+1;currIndex<endIndex;currIndex++)
	{
		/*if (clock()-nTmStart>10*CLOCKS_PER_SEC){
			cout<<"speed: "<<currIndex/10<<" fps"<<endl;
			break;
		}*/
		cout<<"Reading files"<<currIndex<<endl;
		FRAME currFrame=readFrame(currIndex,pd);
		clock_t nTmTmp=clock();
		computeKeyPointsAndDesp(currFrame,detector,descriptor);
		g_count1+=clock()-nTmTmp;
		clock_t nTmTmp2=clock();
		CHECK_RESULT result=checkKeyframes(keyframes.back(),currFrame,globalOptimizer);//match I(currFrame) with the last frame in F(keyframes)
		g_count2+=clock()-nTmTmp2;
		switch(result)
		{
		case NOT_MATCHED:
			//if not matched just jump over
			red(cout)<<"Not enough inliers."<<white<<endl;
			break;
		case TOO_FAR_AWAY:
			//too far so jump
			cout<<red<<"Too far away, may be an error."<<white<<endl;//here one template transforms ostream1 << ostream2 to ostream2(ostream1)
			break;
		case TOO_CLOSE:
			//too close so maybe wrong
			cout<<red<<"Too close, not a keyframe"<<white<<endl;
			break;
		case KEYFRAME:
			cout<<green<<"This is a new keyframe"<<white<<endl;
			//not too far and not too close, which is very very important!!!
			//check the closure
			if (check_loop_closure)
			{
				checkNearbyLoops(keyframes,currFrame,globalOptimizer);
				checkRandomLoops(keyframes,currFrame,globalOptimizer);
			}
			keyframes.push_back(currFrame);
			/*cv::resize(currFrame.rgb,rgbsh,shSize);
			for (int i=0;i<rgbsh.rows;++i){
				for (int j=0;j<rgbsh.cols;++j){
					rgbsh.ptr<uchar>(i)[j*3]=rgbsh.ptr<uchar>(i)[j*3]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
					rgbsh.ptr<uchar>(i)[j*3+1]=rgbsh.ptr<uchar>(i)[j*3+1]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
					rgbsh.ptr<uchar>(i)[j*3+2]=rgbsh.ptr<uchar>(i)[j*3+2]*0.5+0.5*currFrame.depth.ptr<ushort>(i)[j]*255.0/4500;
				}
			}
			cout<<rgbsh.cols<<endl;
			srcMat=rgbsh.clone();
			imshow("Test Color",rgbsh);
			cv::waitKey(1);*/
			break;
		default:
			break;
		}
	}
	cout<<"speed: "<<currIndex-startIndex<<"/"<<(clock()-nTmStart)/CLOCKS_PER_SEC<<" fps"<<endl;
	cout<<"g1: "<<g_count1<<" g2:"<<g_count2<<" g_pnp:"<<g_count<<endl;
	//optimize all edges
	cout<<"optimizing pose grah, vertices: "<<globalOptimizer.vertices().size()<<endl;
	globalOptimizer.save("result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);//we can designate the optimization step number
	globalOptimizer.save("result_after.g2o");
	cout<<"Optimization done."<<endl;

	//splice the point cloud map
	cout<<"saving the point cloud map..."<<endl;
	PointCloud::Ptr output(new PointCloud());//global map
	PointCloud::Ptr tmp(new PointCloud);
	CAMERA_INTRINSIC_PARAMETERS camCM=getDefaultCamera();//get camera calibration matrix

	pcl::VoxelGrid<PointT> voxel;//grid filter, adjust the map resolution
	double gridsize=atof(pd.getData("voxel_grid").c_str());//resolution can be adjusted in parameters.txt
	voxel.setLeafSize(gridsize,gridsize,gridsize);

	pcl::PassThrough<PointT> pass;//the z direction field filter, for the effective depth interval of rgbd camera is limited, delete the too far ones
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0,4.0);//the data where depth is over 4m will be throwed away

	for (size_t i=0;i<keyframes.size();i++)
	{
		//get one frame from g2o
		g2o::VertexSE3* vertex=dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));//change class between base and derived one
		Eigen::Isometry3d pose=vertex->estimate();//the pose matrix of the optimized frame
		PointCloud::Ptr newCloud=image2PointCloud(keyframes[i].rgb,keyframes[i].depth,camCM);//change this frame to PC
		//the following is filtering
		voxel.setInputCloud(newCloud);
		voxel.filter(*tmp);
		pass.setInputCloud(tmp);
		pass.filter(*newCloud);
		//put the changed PC to global map
		pcl::transformPointCloud(*newCloud,*tmp,pose.matrix());//here pose.matrix()=xj=Tj0*x0=Tj0
		*output+=*tmp;
		//tmp->clear();
		newCloud->clear();
	}

	voxel.setInputCloud(output);
	voxel.filter(*tmp);
	//save
	cout<<"point cloud size = "<<tmp->points.size()<<endl;
	pcl::io::savePCDFile("result.pcd",*tmp);
	cout<<"Final map is saved."<<endl;

	for (int i=0;i<loopIndexPairs.size()-1;i+=2){
		cout<<"Random: "<<loopIndexPairs[i]<<" "<<loopIndexPairs[i+1]<<endl;
	}

	/*pcl::visualization::CloudViewer viewer("Viewer");
	viewer.showCloud(cloud);
	while (!viewer.wasStopped()){
	}*/

	//clear PC&G2O
	output->clear();
	tmp->clear();
	globalOptimizer.clear();
	g2o::RobustKernelFactory::destroy();

	//clear KV2 data and return
ENDCAPTURE:
	if (colorfs!=NULL){
		colorfs->Release();
		colorfs=NULL;
	}
	colorfr->Release();
	frameds->Release();
	depthfs->Release();
	depthfr->Release();
	depthfd->Release();
ENDCLOSE:
	pKinect->Close();
	pKinect->Release();
	delete[] pColorSP;
ENDSTOP:
	if (strAns=="Y")
		cv::destroyAllWindows();
	system("pause");

	return 0;
}

CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
	CAMERA_INTRINSIC_PARAMETERS camera;
	ParameterReader pd;
	camera.fx=atof(pd.getData("camera.fx").c_str());
	camera.fy=atof(pd.getData("camera.fy").c_str());
	camera.cx=atof(pd.getData("camera.cx").c_str());
	camera.cy=atof(pd.getData("camera.cy").c_str());
	camera.scale=atof(pd.getData("camera.scale").c_str());

	return camera;
}
FRAME readFrame(int index, ParameterReader& pd)
{
	FRAME f;
	string rgbDir=pd.getData("rgb_dir");
	string depthDir=pd.getData("depth_dir");

	string rgbExt=pd.getData("rgb_extension");
	string depthExt=pd.getData("depth_extension");

	stringstream ss;
	ss<<rgbDir<<index<<rgbExt;
	string filename;
	ss>>filename;
	f.rgb=cv::imread(filename);

	ss.clear();
	filename.clear();
	ss<<depthDir<<index<<depthExt;
	ss>>filename;

	f.depth=cv::imread(filename,-1);//-1 for 16UC1
	f.frameID=index;
	
	return f;
}
FRAME readFrame(int index, ParameterReader& pd,IColorFrameReader* colorfr,IDepthFrameReader* depthfr,
				cv::Size sizeC,cv::Size sizeD,ColorSpacePoint* pColorSP,bool bFirst)
{
	//Create photo matrices
	//rgb picture is the color pic with 8UC3
	//depth is the single channel pic with 16UC1, flags should be -1, meaning that no revision will be done
	cv::Mat rgb(sizeC.height,sizeC.width,CV_8UC4),depth(sizeD.height,sizeD.width,CV_16UC1);//here color must use 4ch as bgra for the kinect output format
	cv::Size shSize=cv::Size(sizeC.width/2,sizeC.height/2);//notice Size uses width first!
	cv::Mat rgbsh(shSize,CV_8UC4),depsh(sizeD.height,sizeD.width,CV_8UC1);

	static clock_t fps_tmstart=clock();
	static int fps_times=0;
	FRAME f;
	f.frameID=-1;
	ColorSpacePoint* pColorSP2=new ColorSpacePoint[sizeD.height*sizeD.width];
	do
	{
		IColorFrame* colorf=nullptr;
		HRESULT hr=colorfr->AcquireLatestFrame(&colorf);//get the latest data
		if (SUCCEEDED(hr)){
			colorf->CopyConvertedFrameDataToArray(sizeC.height*sizeC.width*4,
				reinterpret_cast<BYTE*>(rgb.data),
				ColorImageFormat::ColorImageFormat_Bgra);
		}else{
			cout<<red<<"error in taking rgb photos!!!"<<endl;
			return f;
		}
		if (colorf!=NULL){
			colorf->Release();
			colorf=NULL;
		}
		IDepthFrame* depthf=NULL;
		hr=depthfr->AcquireLatestFrame(&depthf);
		if (SUCCEEDED(hr)){
			depthf->CopyFrameDataToArray(sizeD.height*sizeD.width,
				(UINT16*)depth.data);
			depthf->Release();
		}else{
			cout<<green<<"error in taking depth photos!!!"<<endl;
			return f;
		}
		/*for (int i=0;i<rgb.rows;++i){
			for (int j=0;j<rgb.cols/2;++j){
				for (int k=0;k<4;++k){
					int tmp=rgb.ptr<uchar>(i)[j*4+k];
					rgb.ptr<uchar>(i)[j*4+k]=rgb.ptr<uchar>(i)[(rgb.cols-1-j)*4+k];
					rgb.ptr<uchar>(i)[(rgb.cols-1-j)*4+k]=tmp;
				}
			}
		}
		for (int i=0;i<depth.rows;++i){
			for (int j=0;j<depth.cols/2;++j){
				int tmp=depth.ptr<ushort>(i)[j];
				depth.ptr<ushort>(i)[j]=depth.ptr<ushort>(i)[depth.cols-1-j];
				depth.ptr<ushort>(i)[depth.cols-1-j]=tmp;
			}
			for (int j=0;j<depth.cols;++j)
				if (depth.ptr<ushort>(i)[j]==0);
				//	depth.ptr<ushort>(i)[j]=4500;
				//else depth.ptr<ushort>(i)[j]+=275;
		}*/
		HRESULT hres=pMapper->MapColorFrameToDepthSpace(sizeD.height*sizeD.width,(UINT16*)depth.data,sizeC.height*sizeC.width,(DepthSpacePoint*)pColorSP);
		if (hres==S_OK)
			hres=pMapper->MapDepthFrameToColorSpace(sizeD.height*sizeD.width,(UINT16*)depth.data,sizeD.height*sizeD.width,pColorSP2);
		if (hres!=S_OK){
			cout<<"error_code= "<<hres<<endl;
			system("pause");
		}
		/*for (int i=0;i<sizeD.height*sizeD.width;++i)
			cout<<pColorSP2[i].X<<" ";
		cout<<endl;
		system("pause");
		struct Node{
			int x,y;
			ushort d;
			Node():d(-1){}
		};
		vector<Node> vecColorUsed(sizeC.height*sizeC.width);*/
		depth.convertTo(depsh,CV_8UC1,255.0/4500);
		/*for (int m=0;m<depth.rows;m++)
			for (int n=0;n<depth.cols;n++){
				int mapy=pColorSP2[m*depth.cols+n].Y,mapx=pColorSP2[m*depth.cols+n].X;
				//int min_val=~0u>>1,miny=mapy,minx=mapx;
				//mapy=std::floor(pColorSP2[m*depth.cols+n].Y+0.5);
				//mapx=std::floor(pColorSP2[m*depth.cols+n].X+0.5);
				if (mapy<0||mapy>=rgb.rows||depth.ptr<ushort>(m)[n]==0||
					mapx<0||mapx>=rgb.cols)
				{
					rgbsh.ptr<uchar>(m)[n*4]=rgbsh.ptr<uchar>(m)[n*4+1]=rgbsh.ptr<uchar>(m)[n*4+2]=0;
				}else{
					int index_tmp=mapy*rgb.cols+mapx;
					if (abs(pColorSP[index_tmp].Y-m)<=1&&abs(pColorSP[index_tmp].X-n)<=1){
					//if (vecColorUsed[index_tmp].d<=0||depth.ptr<ushort>(m)[n]!=0&&depth.ptr<ushort>(m)[n]<vecColorUsed[index_tmp].d){
						//if (vecColorUsed[index_tmp].d>0)
						//	rgbsh.ptr<uchar>(vecColorUsed[index_tmp].y)[vecColorUsed[index_tmp].x*4]=rgbsh.ptr<uchar>(vecColorUsed[index_tmp].y)[vecColorUsed[index_tmp].x*4+1]=rgbsh.ptr<uchar>(vecColorUsed[index_tmp].y)[vecColorUsed[index_tmp].x*4+2]=0;
						//vecColorUsed[index_tmp].y=m;vecColorUsed[index_tmp].x=n;
						//vecColorUsed[index_tmp].d=depth.ptr<ushort>(m)[n];
						rgbsh.ptr<uchar>(m)[n*4]=rgb.data[index_tmp*4];//change 4 channels to 3 and use pColorSP[], b
						rgbsh.ptr<uchar>(m)[n*4+1]=rgb.data[index_tmp*4+1];//g
						rgbsh.ptr<uchar>(m)[n*4+2]=rgb.data[index_tmp*4+2];//r
						for (int k=0;k<3;++k){
							int tmp=rgbsh.ptr<uchar>(m)[n*4+k]*0.5+0.5*depsh.ptr<uchar>(m)[n];
							if (tmp>255) tmp=255;
							rgbsh.ptr<uchar>(m)[n*4+k]=tmp;
						}
					}else
						rgbsh.ptr<uchar>(m)[n*4]=rgbsh.ptr<uchar>(m)[n*4+1]=rgbsh.ptr<uchar>(m)[n*4+2]=0;//be kept out
				}
			}*/
		cv::resize(rgb,rgbsh,shSize);
		/*depsh=cv::Mat::zeros(depsh.rows,depsh.cols,CV_8UC1);
		for (int i=0;i<rgbsh.rows;++i){
			int m=i*rgb.rows/rgbsh.rows;
			for (int j=0;j<rgbsh.cols;++j){
				int n=j*rgb.cols/rgbsh.cols;
				int mapy=pColorSP[m*rgb.cols+n].Y,mapx=pColorSP[m*rgb.cols+n].X;
				/*for (int k1=0;k1<2;++k1){
					for (int k2=0;k2<4;++k2){
						mapy=std::floor(pColorSP[m*rgb.cols+n].Y+0.5*k1);
						mapx=std::floor(pColorSP[m*rgb.cols+n].X+0.5*k2);
						if (mapy>=0&&mapy<depth.rows&&mapx>=0&&mapx<depth.cols
							&&depsh.data[mapy*depth.cols+mapx]==0)
								break;
					}
					if (mapy>=0&&mapy<depth.rows&&mapx>=0&&mapx<depth.cols
						&&depsh.data[mapy*depth.cols+mapx]==0)
							break;
				}*//*
				if (mapy<0||mapy>=depth.rows||mapx<0||mapx>=depth.cols)
				{
					for (int k=0;k<3;++k){
						int tmp=rgb.ptr<uchar>(m)[n*4+k]*15.0/255;
						if (tmp>255) tmp=255;
						rgbsh.ptr<uchar>(i)[j*4+k]=tmp;
					}
				}else{
					int index_tmp=mapy*depth.cols+mapx;
					int tmp=((UINT16*)depth.data)[index_tmp]*255.0/4500;
					if (tmp>255) tmp=255;
					depsh.data[index_tmp]=tmp;
					for (int k=0;k<3;++k){
						int tmp=rgb.ptr<uchar>(m)[n*4+k]*(depsh.data[index_tmp]+15.0)/255;
						if (tmp>255) tmp=255;
						rgbsh.ptr<uchar>(i)[j*4+k]=tmp;
					}
				}
			}
		}*/
		cv::flip(rgbsh,rgbsh,1);
		++fps_times;
		static ostringstream oss;
		if (clock()-fps_tmstart>=1000){
			oss.str("");
			oss<<"fps: "<<fps_times*1000/(clock()-fps_tmstart);
			fps_times=0;
			fps_tmstart=clock();
		}
		cv::putText(rgbsh,oss.str(),cv::Point(rgbsh.cols/10,rgbsh.rows/10),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,0,0));
		srcMat=rgbsh.clone();
		imshow("Test Color",rgbsh);
		cv::flip(depsh,depsh,1);
		imshow("Test Depth",depsh);
	}while (cv::waitKey(30)!=VK_ESCAPE&&bFirst);
	delete pColorSP2;

	////f.depth=cv::Mat(sizeD.height,sizeD.width,CV_16UC1);
	//x:(110-860)*2 is effective
	//depth.copyTo(f.depth);
	f.rgb=cv::Mat(shSize,CV_8UC3);
	for (int m=0;m<f.rgb.rows;m++)
		for (int n=0;n<f.rgb.cols;n++){
			int index_tmp=(m*rgb.rows/f.rgb.rows*rgb.cols+n*rgb.cols/f.rgb.cols)*4;
			f.rgb.ptr<uchar>(m)[n*3]=rgb.data[index_tmp];//change 4 channels to 3, b
			f.rgb.ptr<uchar>(m)[n*3+1]=rgb.data[index_tmp+1];//g
			f.rgb.ptr<uchar>(m)[n*3+2]=rgb.data[index_tmp+2];//r
		}
	f.depth=cv::Mat::zeros(f.rgb.rows,f.rgb.cols,CV_16UC1);
	for (int i=0;i<f.rgb.rows;i++){
		int m=i*rgb.rows/f.rgb.rows;
		for (int j=0;j<f.rgb.cols;j++){
			int n=j*rgb.cols/f.rgb.cols;
			if (pColorSP[m*rgb.cols+n].Y<0||pColorSP[m*rgb.cols+n].Y>=depth.rows||
					pColorSP[m*rgb.cols+n].X<0||pColorSP[m*rgb.cols+n].X>=depth.cols)
			{
			}else{
				int index_tmp=(int)pColorSP[m*rgb.cols+n].Y*depth.cols+(int)pColorSP[m*rgb.cols+n].X;
				int tmp=((UINT16*)depth.data)[index_tmp];
				if (tmp>0) tmp+=275;//about 27.5cm depth offset for my kinectv2!
				f.depth.ptr<ushort>(i)[j]=tmp;
			}
		}
	}
	cv::flip(f.rgb,f.rgb,1);
	cv::flip(f.depth,f.depth,1);
	f.frameID=index;
	//imshow("Test Color2",f.rgb);
	//cout<<40<<" 250: "<<pColorSP[40*depth.cols+250].Y<<" "<<pColorSP[40*depth.cols+250].X<<endl;//71,930

	//save the f.rgb & f.depth
	string rgbDir=pd.getData("rgb_dir");
	string depthDir=pd.getData("depth_dir");

	string rgbExt=pd.getData("rgb_extension");
	string depthExt=pd.getData("depth_extension");

	stringstream ss;
	ss<<rgbDir<<index<<rgbExt;
	string filename;
	ss>>filename;
	cv::imwrite(filename,f.rgb);

	ss.clear();
	filename.clear();
	ss<<depthDir<<index<<depthExt;
	ss>>filename;
	cv::imwrite(filename,f.depth);//-1 for 16UC1

	return f;
}
double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
	return fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)))+fabs(cv::norm(tvec));
}

Eigen::Isometry3d Tcurr=Eigen::Isometry3d::Identity();
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
	static ParameterReader pd;
	static int min_inliers=atoi(pd.getData("min_inliers").c_str());
	double max_norm=atof(pd.getData("max_norm").c_str());
	static double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());

	static double max_norm_lp=atof(pd.getData("max_norm_lp").c_str());
	static CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
	//compare f1 & f2
	clock_t nTmTmp=clock();
	RESULT_OF_PNP result=estimateMotion(f1,f2,camera);
	g_count+=clock()-nTmTmp;
	if (result.inliers<min_inliers)//if inliers is not enough, give up currFrame
		return NOT_MATCHED;
	//calculate the motion range
	double norm=normofTransform(result.rvec,result.tvec);
	if (is_loops==false)
	{
		if (norm>=max_norm)
			return TOO_FAR_AWAY;//too far, maybe wrong
	}
	else
	{
		if (norm>=max_norm_lp)//the max norm of loop closure check is larger than normal check but still has a physical limit, where permits far edges
			return TOO_FAR_AWAY;
	}

	if (norm<=keyframe_threshold)
		return TOO_CLOSE;// too adjacent frame
	//add this vertex and the edge connected to lastframe to g2o
	//vertex, only need id
	Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
	if (is_loops==false)//if it's not a loop closure check, the vertex should be added
	{
		g2o::VertexSE3* v=new g2o::VertexSE3();
		v->setId(f2.frameID);
		Tcurr=Tcurr*T.inverse();
		v->setEstimate(Tcurr);//Eigen::Isometry3d::Identity());//
		opti.addVertex(v);
	}
	//edge part
	g2o::EdgeSE3* edge=new g2o::EdgeSE3();
	//connect two vertices of this edge
	edge->vertices()[0]=opti.vertex(f1.frameID);
	edge->vertices()[1]=opti.vertex(f2.frameID);
	g2o::RobustKernel* robustKernel=g2o::RobustKernelFactory::instance()->construct("Cauchy");//if use static, the program will be stopped at the end for unregistered problem
	edge->setRobustKernel(robustKernel);//relieve the false positive problem, avoid a few wrong edges influence the total result
	//info matrix
	Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();
	//info matrix is the inverse of the covariance matrix, showing our preestimation of the edge accuracy
	//for pose is 6D, info matrix is 6*6, suppose the accuracy of position and angle is 0.1 and independently
	//then covariance is the matrix with diagonal 0.01, info matrix will be 100
	information(0,0)=information(1,1)=information(2,2)=100;
	information(3,3)=information(4,4)=information(5,5)=100;
	//can also set the angle larger to show the estimation of angle is more accurate
	edge->setInformation(information);
	//the estimation of edge is the result of pnp solution
	edge->setMeasurement(T.inverse());//for xi=Tij*xj so Tij means the inverse of Tji
	//add this edge to the pose graph
	opti.addEdge(edge);
	return KEYFRAME;
}
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti)
{
	static ParameterReader pd;
	static int nearby_loops=atoi(pd.getData("nearby_loops").c_str());

	//just compare currFrame I with the last m frames in F
	if (frames.size()<=nearby_loops)
	{
		//no enough keyframes, check everyone
		for (size_t i=0;i<frames.size();i++)
		{
			checkKeyframes(frames[i],currFrame,opti,true);
		}
	}
	else
	{
		//check the nearest ones
		for (size_t i=frames.size()-nearby_loops;i<frames.size();i++)
		{
			checkKeyframes(frames[i],currFrame,opti,true);
		}
	}
}
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti)
{
	static ParameterReader pd;
	static int random_loops=atoi(pd.getData("random_loops").c_str());
	srand((unsigned int)time(NULL));
	//randomly get some frames to check
	if (frames.size()<=random_loops)
	{
		for (size_t i=0;i<frames.size();i++)
		{
			checkKeyframes(frames[i],currFrame,opti,true);
		}
	}
	else
	{
		//randomly check loops
		for (int i=0;i<random_loops;i++)
		{
			int index=rand()%frames.size();
			if (checkKeyframes(frames[index],currFrame,opti,true)==KEYFRAME){
				loopIndexPairs.push_back(index);
				loopIndexPairs.push_back(currFrame.frameID);
			}
		}
	}
}
