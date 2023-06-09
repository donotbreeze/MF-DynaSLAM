/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef FRAME_H
#define FRAME_H

//#define SAVE_TIMES

#include<vector>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "ImuTypes.h"
#include "ORBVocabulary.h"

#include <mutex>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3 {
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class MapPoint;
    class KeyFrame;
    class ConstraintPoseImu;
    class GeometricCamera;
    class ORBextractor;

    class Frame {
    public:
        // This public was all added by MF
        // 计算道路3d点坐标和计算一个对象距离
        void Calculroad3dpointandcalculdistance(const cv::Mat &imroadMaskleft, std::vector<cv::KeyPoint> allpoints, std::vector<float> mvroadDepth, const cv::Mat &imleft);

        // 计算道路3d点坐标和所有对象点
        void Calculroad3dpointandcalculalldistance(const cv::Mat &imroadMaskleft, std::vector<cv::KeyPoint> allpoints, std::vector<float> mvroadDepth, const cv::Mat &imleft);

        // 计算最下距离点坐标
        void CalculMinpoint(const cv::Mat &imMask, cv::Point2f &orginpoint, cv::Point2f &minpoint);

        //　计算掩码值对应的在全局的像素面积(数量)和在ROI(右相机视野)的数量
        void Calculmaskarea(const cv::Mat &imroadMaskleft, int &maskvalue, int &allarea, int &roiarea, cv::Point2f &allminpoint, cv::Point2f &roiminpoint);

        // 计算经过Ｈ矩阵转换的点位置
        void TransportpointforH(cv::Point2f &inpoint, cv::Point2f &outpoint, cv::Mat &H);

        // 选择距离函数
        void Choicedistance(float &visiondistance, float &perspectdistance, int &allarea, int &roiarea, float &bestdistance);

        // 选择距离函数
        void Choicebetterdistance(float &visiondistance, float &perspectdistance, float &preimgdistance, int &allarea, int &roiarea, float &bestdistance);

        //　读取mask非像素值非255和0的种类数
        void Findmaskclass(const cv::Mat &maskimg, int &classnums);

        // 输入两个点，调用函数TransportpointforH映射，如果距离为负数，则循环使用相邻点计算，要考虑边界调点
        void Findtrueperspectdistance(cv::Point2f &inpoint, cv::Point2f &outpoint, cv::Mat &H, const cv::Mat &maskimg, int &times, float &distance);

        void undistComputeStereoMatches();
        // 不同焦距的匹配方法
        void ComputeDifferentfocalStereoMatches();

        // dyna保存一份
        std::vector<cv::KeyPoint> mvKeysdyna, mvKeysRightdyna;
        std::vector<cv::KeyPoint> refermvKeysdyna, refermvKeysRightdyna;
        std::vector<cv::KeyPoint> mvKeysUndyna, mvKeysRightUndyna;

        // 帧mask类别数量
        int maskclassnums;

        // Corresponding stereo coordinate and depth for each keypoint.
        std::vector<MapPoint *> mvpMapPoints;
        // "Monocular" keypoints have a negative value.
        std::vector<float> mvuRight;
        std::vector<float> mvuLeft;
        std::vector<float> mvDepth;
        //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        std::vector<float> LeftIdtoRightId;
        std::vector<float> RightIdToLeftId;

        // dyam_slam将对原容器进行复制值，在对原容器是动态点的对象进行点的去除，用复制后特征容器对各项匹配和可视化等等操作
        // 其目的就是剔除原来容器中是动态对象的点，并且还能记录动态点和全图的匹配关系，这一部分将在tracking线程中完成
        std::vector<float> mvuRightdyna;
        std::vector<float> mvuLeftdyna;
        std::vector<float> mvDepthdyna;
        //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        std::vector<float> LeftIdtoRightIddyna;
        std::vector<float> RightIdToLeftIddyna;

        // 记录每个mask的距离,每个mask对应一个距离，用于去除动态对象
        vector<vector<float> > maskObjDistance;

        // 道路3d点
        vector<cv::Point3f> road3dpoint;
        // 模拟映射平面点
        vector<cv::Point2f> plane2dpoint;

        // mask 图像点
        vector<cv::Point2f> goodmaskminpoint;


        //上一帧到当前帧的本质矩阵Ｅ和基础矩阵
        cv::Mat Ecur, Fcur;


    public:
        Frame();

        float max1(float &a, float &b);
        // Copy constructor.
        Frame(const Frame &frame);

        // Constructor for stereo cameras.
        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF = static_cast<Frame *>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());
        // Constructor for stereo cameras with mask.
        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const cv::Mat &imMaskleft, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF = static_cast<Frame *>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());
        // Constructor for RGB-D cameras.
        Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame *pPrevF = static_cast<Frame *>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

        // Constructor for Monocular cameras.
        Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, GeometricCamera *pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame *pPrevF = static_cast<Frame *>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());


        // Destructor
        // ~Frame();

        // Extract ORB on the image. 0 for left image and 1 for right image.
        void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);

        // Compute Bag of Words representation.
        void ComputeBoW();

        // Set the camera pose. (Imu pose is not modified!)
        void SetPose(cv::Mat Tcw);
        void GetPose(cv::Mat &Tcw);

        // Set IMU velocity
        void SetVelocity(const cv::Mat &Vwb);

        // Set IMU pose and velocity (implicitly changes camera pose)
        void SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb);


        // Computes rotation, translation and camera center matrices from the camera pose.
        void UpdatePoseMatrices();

        // Returns the camera center.
        inline cv::Mat GetCameraCenter() {
            return mOw.clone();
        }

        // Returns inverse of rotation
        inline cv::Mat GetRotationInverse() {
            return mRwc.clone();
        }

        cv::Mat GetImuPosition();
        cv::Mat GetImuRotation();
        cv::Mat GetImuPose();

        void SetNewBias(const IMU::Bias &b);

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

        bool ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v);

        cv::Mat inRefCoordinates(cv::Mat pCw);

        // Compute the cell of a keypoint (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1, const bool bRight = false) const;

        // Search a match for each keypoint in the left image to a keypoint in the right image.
        // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
        void ComputeStereoMatches();


        // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
        void ComputeStereoFromRGBD(const cv::Mat &imDepth);

        // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
        cv::Mat UnprojectStereo(const int &i);

        ConstraintPoseImu *mpcpi;

        bool imuIsPreintegrated();
        void setIntegrated();

        cv::Mat mRwc;
        cv::Mat mOw;
    public:

        // Vocabulary used for relocalization.
        ORBVocabulary *mpORBvocabulary;

        // Feature extractor. The right is used only in the stereo case.
        ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

        // Frame timestamp.
        double mTimeStamp;

        // Calibration matrix and OpenCV distortion parameters.
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        // Stereo baseline multiplied by fx.
        float mbf;

        // Stereo baseline in meters.
        float mb;

        // Threshold close/far points. Close points are inserted from 1 view.
        // Far points are inserted as in the monocular case from 2 views.
        float mThDepth;

        // Number of KeyPoints.
        int N;

        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
        std::vector<cv::KeyPoint> refermvKeys, refermvKeysRight;
        std::vector<cv::KeyPoint> mvKeysUn, mvKeysRightUn;


        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // ORB descriptor, each row associated to a keypoint.
        cv::Mat mDescriptors, mDescriptorsRight;

        // MapPoints associated to keypoints, NULL pointer if no association.
        // Flag to identify outlier associations.观测不到Map中的3D点
        std::vector<bool> mvbOutlier;
        int mnCloseMPs;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];


        // Camera pose.
        cv::Mat mTcw;


        // IMU linear velocity
        cv::Mat mVw;

        cv::Mat mPredRwb, mPredtwb, mPredVwb;
        IMU::Bias mPredBias;

        // IMU bias
        IMU::Bias mImuBias;

        // Imu calibration
        IMU::Calib mImuCalib;

        // Imu preintegration from last keyframe
        IMU::Preintegrated *mpImuPreintegrated;
        KeyFrame *mpLastKeyFrame;

        // Pointer to previous frame
        Frame *mpPrevFrame;
        IMU::Preintegrated *mpImuPreintegratedFrame;

        // Current and Next Frame id.
        static long unsigned int nNextId;
        long unsigned int mnId;

        // Reference Keyframe.
        KeyFrame *mpReferenceKF;

        // Scale pyramid info.
        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        vector<float> mvScaleFactors;
        vector<float> mvInvScaleFactors;
        vector<float> mvLevelSigma2;
        vector<float> mvInvLevelSigma2;

        // Undistorted Image Bounds (computed once).
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;

        map<long unsigned int, cv::Point2f> mmProjectPoints;
        map<long unsigned int, cv::Point2f> mmMatchedInImage;

        string mNameFile;

        int mnDataset;

        double mTimeStereoMatch;
        double mTimeORB_Ext;
        //　原来是私函数
        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();
        void DynaAssignFeaturesToGrid();

    private:

        // Undistort keypoints given OpenCV distortion parameters.
        // Only for the RGB-D case. Stereo must be already rectified!
        // (called in the constructor).
        void UndistortKeyPoints();
        void UndistortLeftKeyPoints();
        void UndistortRightKeyPoints();
        // Computes image bounds for the undistorted image (called in the constructor).
        void ComputeImageBounds(const cv::Mat &imLeft);

        // Rotation, translation and camera center
        cv::Mat mRcw;
        cv::Mat mtcw;
        //==mtwc

        bool mbImuPreintegrated;

        std::mutex *mpMutexImu;

    public:
        GeometricCamera *mpCamera, *mpCamera2;

        //Number of KeyPoints extracted in the left and right images
        int Nleft, Nright;
        //Number of Non Lapping Keypoints
        int monoLeft, monoRight;

        //For stereo matching
        std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

        //For stereo fisheye matching
        static cv::BFMatcher BFmatcher;

        //Triangulated stereo observations using as reference the left camera. These are
        //computed during ComputeStereoFishEyeMatches
        std::vector<cv::Mat> mvStereo3Dpoints;

        //Grid for the right image
        std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        cv::Mat mTlr, mRlr, mtlr, mTrl;

        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, GeometricCamera *pCamera2, cv::Mat &Tlr, Frame *pPrevF = static_cast<Frame *>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

        //Stereo fisheye
        void ComputeStereoFishEyeMatches();

        bool isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight = false);

        cv::Mat UnprojectStereoFishEye(const int &i);

        cv::Mat imgLeft, imgRight;
        cv::Mat maskleft;
        cv::Mat premaskleft;//膨胀之前的

        void PrintPointDistribution() {
            int left = 0, right = 0;
            int Nlim = (Nleft != -1) ? Nleft : N;
            for(int i = 0; i < N; i++) {
                if(mvpMapPoints[i] && !mvbOutlier[i]) {
                    if(i < Nlim) {
                        left++;
                    } else {
                        right++;
                    }
                }
            }
            cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << endl;
        }
    };

}// namespace ORB_SLAM

#endif // FRAME_H
