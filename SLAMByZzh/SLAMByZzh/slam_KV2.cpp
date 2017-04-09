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

//when mouse clicked, show the coordinate
IplImage* src=0;
void on_mouse(int event,int x,int y,int flags,void* ustc){
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1,1,0,2,CV_AA);
	if (event==CV_EVENT_LBUTTONDOWN){
		CvPoint pt=cvPoint(x*4/5,y*540/695);//???
		char temp[16];
		sprintf(temp,"(%d,%d)",x,y);
		cvPutText(src,temp,pt,&font,cvScalar(255,255,255,0));
		cvCircle(src,pt,4,cvScalar(255,0,0,0),CV_FILLED,CV_AA);
		//cv::Mat srcMat=cv::Mat(src);
		cvShowImage("Test Color",src);//cv::imshow("Test Color",srcMat);
		printf("%d,%d, Color\n",x,y);
	}
}
void on_mouse2(int event,int x,int y,int flags,void* ustc){
	if (event==CV_EVENT_LBUTTONDOWN){
		printf("%d,%d, Depth\n",x,y);
	}
}

vector<int> loopIndexPairs;

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

	//Create windows
	//API:http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#cv2.imread
	cv::namedWindow("Test Color");//call a window
	cv::namedWindow("Test Depth");
	cvSetMouseCallback("Test Color",on_mouse);
	cvSetMouseCallback("Test Depth",on_mouse2);
	//get pColorSP for KV2
	int count=hei2*wid2;
	ColorSpacePoint* pColorSP=new ColorSpacePoint[count];
	ICoordinateMapper* pMapper;
	pKinect->get_CoordinateMapper(&pMapper);
	CameraIntrinsics camIntr;
	pMapper->GetDepthCameraIntrinsics(&camIntr);
	cout<<camIntr.FocalLengthX<<" "<<camIntr.FocalLengthY<<" "
		<<camIntr.PrincipalPointX<<" "<<camIntr.PrincipalPointY<<" "
		<<camIntr.RadialDistortionSecondOrder<<" "<<camIntr.RadialDistortionFourthOrder
		<<" "<<camIntr.RadialDistortionSixthOrder<<endl;
#ifdef _DEBUG
	UINT16* pDepthData=new UINT16[hei2*wid2];
	/*for (int i=0;i<hei2;i++)
		for (int j=0;j<wid2;j++)
		{
			pDepthData[i*wid2+j]=((UINT16*)depth.data)[i*wid2+j];//depth.ptr<ushort>(i)[j];
			cout<<pDepthData[i*wid2+j]<<endl;
		}*/
	pMapper->MapDepthFrameToColorSpace(count,pDepthData,count,pColorSP);//(UINT16*)depth.data
	delete[] pDepthData;
	ofstream fout("pColorSPd.txt");
	for (int i=0;i<hei2;i++)
	{
		for (int j=0;j<wid2;j++)
		{
			fout<<pColorSP[i*wid2+j].X<<" "<<pColorSP[i*wid2+j].Y<<" ";
		}
		fout<<endl;
	}
	fout.close();//fout<<flush;
#else
	ifstream fin("pColorSPd.txt");
	for (int i=0;i<hei2;i++)
	{
		for (int j=0;j<wid2;j++)
		{
			fin>>pColorSP[i*wid2+j].X>>pColorSP[i*wid2+j].Y;
		}
	}
	fin.close();
#endif


	//SLAM
	ParameterReader pd;//get the basic slam information
	vector<FRAME> keyframes;//keyframes array
	//save SLAM images from kinectv2
	int currIndex=0;
	cout<<"Do you want to take photos?Y or N"<<endl;
	string strAns;
	cin>>strAns;
	if (strAns=="Y"){
		do
		{
			//Sleep(300);
			cout<<"Reading files"<<currIndex<<endl;
			FRAME f=readFrame(currIndex,pd,colorfr,depthfr,
			cv::Size(width,height),cv::Size(wid2,hei2),pColorSP,!currIndex);//read the KV2 to get currFrame
			if (f.frameID!=-1)
				currIndex++;
		}while (currIndex<400);
	}else{
		cv::destroyAllWindows();
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

CYCLE:
	//the id of last frame is already recorded in FRAME structure
	for (currIndex=startIndex+1;currIndex<endIndex;currIndex++)
	{
		cout<<"Reading files"<<currIndex<<endl;
		FRAME currFrame=readFrame(currIndex,pd);
		computeKeyPointsAndDesp(currFrame,detector,descriptor);
		CHECK_RESULT result=checkKeyframes(keyframes.back(),currFrame,globalOptimizer);//match I(currFrame) with the last frame in F(keyframes)
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
			break;
		default:
			break;
		}
	}

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
	tmp->height=1;tmp->width=tmp->points.size();tmp->is_dense=false;//if don't initialize tmp and save it as pcd, the program will be stopped at the end
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

	FRAME f;
	f.frameID=-1;
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
		depth.convertTo(depsh,CV_8UC1,255.0/4500);
		cv::resize(rgb,rgbsh,shSize);
		src=&IplImage(rgbsh);
		imshow("Test Color",rgbsh);
		imshow("Test Depth",depsh);
	}while (cv::waitKey(30)!=VK_ESCAPE&&bFirst);

	//f.depth=cv::Mat(sizeD.height,sizeD.width,CV_16UC1);
	depth.copyTo(f.depth);
	f.rgb=cv::Mat(sizeD.height,sizeD.width,CV_8UC3);
	for (int m=0;m<depth.rows;m++)
		for (int n=0;n<depth.cols;n++){
			if (pColorSP[m*depth.cols+n].Y<0||pColorSP[m*depth.cols+n].Y>=rgb.rows||
					pColorSP[m*depth.cols+n].X<0||pColorSP[m*depth.cols+n].X>=rgb.cols)
			{
				f.rgb.ptr<uchar>(m)[n*3]=f.rgb.ptr<uchar>(m)[n*3+1]=f.rgb.ptr<uchar>(m)[n*3+2]=0;
			}else{
				f.rgb.ptr<uchar>(m)[n*3]=rgb.data[((int)pColorSP[m*depth.cols+n].Y*rgb.cols+(int)pColorSP[m*depth.cols+n].X)*4];//change 4 channels to 3 and use pColorSP[], b
				f.rgb.ptr<uchar>(m)[n*3+1]=rgb.data[((int)pColorSP[m*depth.cols+n].Y*rgb.cols+(int)pColorSP[m*depth.cols+n].X)*4+1];//g
				f.rgb.ptr<uchar>(m)[n*3+2]=rgb.data[((int)pColorSP[m*depth.cols+n].Y*rgb.cols+(int)pColorSP[m*depth.cols+n].X)*4+2];//r
			}
		}
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

CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
	static ParameterReader pd;
	static int min_inliers=atoi(pd.getData("min_inliers").c_str());
	double max_norm=atof(pd.getData("max_norm").c_str());
	static double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());

	static double max_norm_lp=atof(pd.getData("max_norm_lp").c_str());
	static CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
	static g2o::RobustKernel* robustKernel=g2o::RobustKernelFactory::instance()->construct("Cauchy");
	//compare f1 & f2
	RESULT_OF_PNP result=estimateMotion(f1,f2,camera);
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
	if (is_loops==false)//if it's not a loop closure check, the vertex should be added
	{
		g2o::VertexSE3* v=new g2o::VertexSE3();
		v->setId(f2.frameID);
		v->setEstimate(Eigen::Isometry3d::Identity());
		opti.addVertex(v);
	}
	//edge part
	g2o::EdgeSE3* edge=new g2o::EdgeSE3();
	//connect two vertices of this edge
	edge->vertices()[0]=opti.vertex(f1.frameID);
	edge->vertices()[1]=opti.vertex(f2.frameID);
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
	Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
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
