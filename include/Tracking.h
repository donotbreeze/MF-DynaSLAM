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

#pragma once
#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include"Viewer.h"
#include"FrameDrawer.h"
#include"Atlas.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"

#include "GeometricCamera.h"

#include <mutex>
#include <unordered_set>
using namespace cv;

namespace ORB_SLAM3 {

    class Viewer;
    class FrameDrawer;
    class Atlas;
    class LocalMapping;
    class LoopClosing;
    class System;

    class Tracking {
    public:
        vector<Vec3b> MaskColor = {
            Vec3b(238, 99, 99),
            Vec3b(255, 105, 180),
            Vec3b(144, 238, 144),
            Vec3b(255, 225, 255),
            Vec3b(255, 174, 185),
            Vec3b(0, 255, 127),
            Vec3b(0, 255, 255),
            Vec3b(255, 182, 193),
            Vec3b(220, 20, 60),
            Vec3b(255, 0, 255),
            Vec3b(148, 0, 211),
            Vec3b(138, 43, 226),
            Vec3b(65, 105, 225),
            Vec3b(30, 144, 255),
            Vec3b(0, 191, 255),
            Vec3b(176, 224, 230),
            Vec3b(175, 238, 238),
            Vec3b(0, 255, 255),
            Vec3b(0, 139, 139),
            Vec3b(64, 224, 208),
            Vec3b(0, 250, 154),
            Vec3b(46, 139, 87),
            Vec3b(152, 251, 152),
            Vec3b(127, 255, 0),
            Vec3b(255, 255, 0),
            Vec3b(255, 215, 0),
            Vec3b(184, 134, 11),
            Vec3b(255, 165, 0),
            Vec3b(255, 228, 196),
            Vec3b(255, 69, 0),
            Vec3b(255, 0, 0),
            Vec3b(220, 220, 220),
            Vec3b(152, 245, 255),
            Vec3b(84, 255, 159),
            Vec3b(152, 251, 152 ),
            Vec3b( 	255, 106, 106 ),
            Vec3b( 	255, 127, 80),
            Vec3b( 	153, 50, 204 ),
            Vec3b(238, 18, 137 ),
            Vec3b(191, 239, 255 ),
            Vec3b(0, 0, 238 )
        };

    public:
        Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas,
                 KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, const string &_nameSeq = std::string());

        ~Tracking();
        // 前后两帧mask之间内的匹配信息，前一帧匹配当前帧
        std::vector<float> FrontbackMatches;
        // 前后两帧mask之间内的匹配信息，但是是去除动态对象点的匹配
        std::vector<float> FrontbackMatchesdyna;
        // 前后两帧mask之间外的匹配信息，前一帧匹配当前帧
        std::vector<float> FrontbackoutMatches;

        // 计算三维向量v的反对称矩阵
        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
        // 三角化
        void Triangulatelast2cur(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        // 前后关键两帧之间mask的匹配信息，前一帧匹配当前帧
        std::vector<float> FrontbackKeyMatches;
        // 前后关键两帧之间mask之外的匹配信息，前一帧匹配当前帧
        std::vector<float> FrontbackoutKeyMatches;

        //颜色
        void AddColor(Mat &imMask, Mat &outmat);

        float CalcullowerThaverdis(vector<float> &pointsdistance, float &th);
        float Calcuaverdistance(vector<float> &savedistance);
        float Calcuaverdistance(vector<vector<float> >  &savedistance, int &classnum);
        // 计算本质矩阵E和基础矩阵F
        cv::Mat CalculMatEF();
        // 处理vector<vector<int> > curMasknum2frontMasknums中筛选一一对应问题
        void ChoicecurMaskOnebyOnefrontMask(vector<vector<int> > &curMasknum2frontMasknums, vector<vector<int> > &maskNumsmatch);
        void CurrentSpeed(Frame &cur, Frame &last, vector<vector<int> > maskNumsmatch);
        // 前后帧匹配
        void Frontandbackframematching();
        // 前后关键帧帧匹配
        void FrontandbackKeyframematching();

        // Preprocess the input and call Track(). Extract features and performs stereo matching.
        cv::Mat GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename);
        cv::Mat GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const cv::Mat &imRectmaskleft, const double &timestamp, string filename);
        cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename);
        cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
        // cv::Mat GrabImageImuMonocular(const cv::Mat &im, const double &timestamp);

        void GrabImuData(const IMU::Point &imuMeasurement);

        void SetLocalMapper(LocalMapping *pLocalMapper);
        void SetLoopClosing(LoopClosing *pLoopClosing);
        void SetViewer(Viewer *pViewer);
        void SetStepByStep(bool bSet);

        // Load new settings
        // The focal lenght should be similar or scale prediction will fail when projecting points
        void ChangeCalibration(const string &strSettingPath);

        // Use this function if you have deactivated local mapping and you only want to localize the camera.
        void InformOnlyTracking(const bool &flag);

        void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame);
        KeyFrame *GetLastKeyFrame() {
            return mpLastKeyFrame;
        }

        void CreateMapInAtlas();
        std::mutex mMutexTracks;

        //--
        void NewDataset();
        int GetNumberDataset();
        int GetMatchesInliers();
    public:

        // Tracking states
        enum eTrackingState {
            SYSTEM_NOT_READY = -1,
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            OK = 2,
            RECENTLY_LOST = 3,
            LOST = 4,
            OK_KLT = 5
        };

        eTrackingState mState;
        eTrackingState mLastProcessedState;

        // Input sensor
        int mSensor;

        // Current Frame
        Frame mCurrentFrame;
        Frame mLastFrame;
        // save lastframe..
        bool bmcur = false, bMFlast = false;
        // cache Frame
        Frame MFLastFrame0, MFLastFrame1, MFLastFrame2, MFLastFrame3, MFLastFrame4;
        // 用帧的属性记录关键帧
        Frame MFLast0KeyFrame, MFLast1KeyFrame;

        cv::Mat mImGray;
        cv::Mat mImmaskleft;
        // Initialization Variables (Monocular)
        std::vector<int> mvIniLastMatches;
        std::vector<int> mvIniMatches;
        std::vector<cv::Point2f> mvbPrevMatched;
        std::vector<cv::Point3f> mvIniP3D;
        Frame mInitialFrame;

        // Lists used to recover the full camera trajectory at the end of the execution.
        // Basically we store the reference keyframe for each frame and its relative transformation
        list<cv::Mat> mlRelativeFramePoses;
        list<KeyFrame *> mlpReferences;
        list<double> mlFrameTimes;
        list<bool> mlbLost;

        // frames with estimated pose
        int mTrackedFr;
        bool mbStep;

        // True if local mapping is deactivated and we are performing only localization
        bool mbOnlyTracking;

        void Reset(bool bLocMap = false);
        void ResetActiveMap(bool bLocMap = false);

        float mMeanTrack;
        bool mbInitWith3KFs;
        double t0; // time-stamp of first read frame
        double t0vis; // time-stamp of first inserted keyframe
        double t0IMU; // time-stamp of IMU initialization


        vector<MapPoint *> GetLocalMapMPS();


        //TEST--
        cv::Mat M1l, M2l;
        cv::Mat M1r, M2r;

        bool mbWriteStats;

    protected:

        // Main tracking function. It is independent of the input sensor.
        void Track();

        // Map initialization for stereo and RGB-D
        void StereoInitialization();

        // Map initialization for monocular
        void MonocularInitialization();
        void CreateNewMapPoints();

        //　保留动态区域的ｍａｓｋ
        Mat SaveDynaMask(vector<bool>& dynamaskerea,Mat& maskimg);
        void readWriteFile(vector<bool> &dynamaskerea,int index);

        // 使用距离判断是否是动态
        bool WhetherDynawithDistance(vector<vector<int> > &maskMatch, int &num, vector<vector<float> > &curDis, vector<vector<float> > &frontDis, float &move);
        bool WhetherDynawithDistance(vector<vector<int> > &maskMatch, int &num, vector<vector<float> > &curDis, vector<vector<float> > &frontDis, vector<vector<int> > &maskbord, float &move);
        // 计算mask边界
        void Calculmaskborder(cv::Mat &maskborder, int &maskclassnums, vector<vector<int> > &border) ;
        // 膨胀掩码
        cv::Mat dilatemask(cv::Mat &mask);
        // 计算ｆ*vec(x1`*x2)
        float CalculFPP(cv::Mat &F, cv::Point2f &p0, cv::Point2f &p1);

        // 普通帧计算T矩阵
        cv::Mat ComputeframeT12(Frame *pKF1, Frame *pKF2);
        // 获取普通帧Ｆ
        cv::Mat ComputeframeF12(Frame *pKF1, Frame *pKF2);
        // 普通帧计算E矩阵
        cv::Mat ComputeframeE12(Frame *pKF1, Frame *pKF2);

        cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
        //计算点距离
        float Calcultwopointdistance(cv::Point2f &pfront, cv::Point2f &pafter);
        float CalculEpipolardistance(cv::Point2f &p2, cv::Point2f &p1, cv::Mat &F);
        float ChoiceoneEpipolardistance(vector<float> &pointsdistance, float &th);

        void CreateInitialMapMonocular();

        void CheckReplacedInLastFrame();
        bool TrackReferenceKeyFrame();
        void UpdateLastFrame();
        bool TrackWithMotionModel();
        bool PredictStateIMU();

        bool Relocalization();

        void UpdateLocalMap();
        void UpdateLocalPoints();
        void UpdateLocalKeyFrames();

        bool TrackLocalMap();
        bool TrackLocalMap_old();
        void SearchLocalPoints();

        bool NeedNewKeyFrame();
        void CreateNewKeyFrame();

        // Perform preintegration from last frame
        void PreintegrateIMU();

        // Reset IMU biases and compute frame velocity
        void ResetFrameIMU();
        void ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx,  float &bwy, float &bwz);
        void ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax,  float &bay, float &baz);


        bool mbMapUpdated;

        // Imu preintegration from last frame
        IMU::Preintegrated *mpImuPreintegratedFromLastKF;

        // Queue of IMU measurements between frames
        std::list<IMU::Point> mlQueueImuData;

        // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
        std::vector<IMU::Point> mvImuFromLastFrame;
        std::mutex mMutexImuQueue;

        // Imu calibration parameters
        IMU::Calib *mpImuCalib;

        // Last Bias Estimation (at keyframe creation)
        IMU::Bias mLastBias;

        // In case of performing only localization, this flag is true when there are no matches to
        // points in the map. Still tracking will continue if there are enough matches with temporal points.
        // In that case we are doing visual odometry. The system will try to do relocalization to recover
        // "zero-drift" localization to the map.
        bool mbVO;

        //Other Thread Pointers
        LocalMapping *mpLocalMapper;
        LoopClosing *mpLoopClosing;

        //ORB
        ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
        ORBextractor *mpIniORBextractor;

        //BoW
        ORBVocabulary *mpORBVocabulary;
        KeyFrameDatabase *mpKeyFrameDB;

        // Initalization (only for monocular)
        Initializer *mpInitializer;
        bool mbSetInit;

        //Local Map
        KeyFrame *mpReferenceKF;
        std::vector<KeyFrame *> mvpLocalKeyFrames;
        std::vector<MapPoint *> mvpLocalMapPoints;

        // System
        System *mpSystem;

        //Drawers
        Viewer *mpViewer;
        FrameDrawer *mpFrameDrawer;
        MapDrawer *mpMapDrawer;
        bool bStepByStep;

        //Atlas
        Atlas *mpAtlas;

        //Calibration matrix
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;

        //New KeyFrame rules (according to fps)
        int mMinFrames;
        int mMaxFrames;

        int mnFirstImuFrameId;
        int mnFramesToResetIMU;

        // Threshold close/far points
        // Points seen as close by the stereo/RGBD sensor are considered reliable
        // and inserted from just one frame. Far points requiere a match in two keyframes.
        float mThDepth;

        // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
        float mDepthMapFactor;

        //Current matches in frame
        int mnMatchesInliers;

        //Last Frame, KeyFrame and Relocalisation Info
        //上一关键帧
        KeyFrame *mpLastKeyFrame;
        //上一个关键帧的ID
        unsigned int mnLastKeyFrameId;
        //上一次重定位的那一帧的ID
        unsigned int mnLastRelocFrameId;
        double mTimeStampLost;
        double time_recently_lost;


        unsigned int mnFirstFrameId;
        unsigned int mnInitialFrameId;
        unsigned int mnLastInitFrameId;

        bool mbCreatedMap;


        //Motion Model
        cv::Mat mVelocity;

        //Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;

        list<MapPoint *> mlpTemporalPoints;

        //int nMapChangeIndex;

        int mnNumDataset;

        ofstream f_track_stats;

        ofstream f_track_times;
        double mTime_PreIntIMU;
        double mTime_PosePred;
        double mTime_LocalMapTrack;
        double mTime_NewKF_Dec;

        GeometricCamera *mpCamera, *mpCamera2;

        int initID, lastID;

        cv::Mat mTlr;

    public:
        cv::Mat mImRight;
    };

} //namespace ORB_SLAM

#endif

