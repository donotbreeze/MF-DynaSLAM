/**
* This file is part of MF-DynaSLAM
*
* 2023 Mingchi Feng, Xuan Yi, Chengnan Li, Jia Zheng.
*
* MF-DynaSLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* MF-DynaSLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with MF-DynaSLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/



#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <algorithm>
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Initializer.h"
#include"G2oTypes.h"
#include"Optimizer.h"
#include"PnPsolver.h"
#include"Points.h"
#include<iostream>
#include"Drawkp.h"
#include<mutex>
#include<chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>


using namespace std;

namespace ORB_SLAM3 {


    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, const string &_nameSeq):
        mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
        mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
        mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
        mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
        mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr) {
        // Load camera parameters from settings file
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        cv::Mat DistCoef = cv::Mat::zeros(4, 1, CV_32F);
        string sCameraName = fSettings["Camera.type"];
        if(sCameraName == "PinHole") {
            float fx = fSettings["Camera.fx"];
            float fy = fSettings["Camera.fy"];
            float cx = fSettings["Camera.cx"];
            float cy = fSettings["Camera.cy"];
            vector<float> vCamCalib{fx, fy, cx, cy};
            mpCamera = new Pinhole(vCamCalib);
            mpAtlas->AddCamera(mpCamera);
            DistCoef.at<float>(0) = fSettings["Camera.k1"];
            DistCoef.at<float>(1) = fSettings["Camera.k2"];
            DistCoef.at<float>(2) = fSettings["Camera.p1"];
            DistCoef.at<float>(3) = fSettings["Camera.p2"];
        }
        if(sCameraName == "KannalaBrandt8") {
            float fx = fSettings["Camera.fx"];
            float fy = fSettings["Camera.fy"];
            float cx = fSettings["Camera.cx"];
            float cy = fSettings["Camera.cy"];
            float K1 = fSettings["Camera.k1"];
            float K2 = fSettings["Camera.k2"];
            float K3 = fSettings["Camera.k3"];
            float K4 = fSettings["Camera.k4"];
            vector<float> vCamCalib{fx, fy, cx, cy, K1, K2, K3, K4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpAtlas->AddCamera(mpCamera);
            if(sensor == System::STEREO || sensor == System::IMU_STEREO) {
                //Right camera
                fx = fSettings["Camera2.fx"];
                fy = fSettings["Camera2.fy"];
                cx = fSettings["Camera2.cx"];
                cy = fSettings["Camera2.cy"];
                K1 = fSettings["Camera2.k1"];
                K2 = fSettings["Camera2.k2"];
                K3 = fSettings["Camera2.k3"];
                K4 = fSettings["Camera2.k4"];
                cout << endl << "Camera2 Parameters: " << endl;
                cout << "- fx: " << fx << endl;
                cout << "- fy: " << fy << endl;
                cout << "- cx: " << cx << endl;
                cout << "- cy: " << cy << endl;
                vector<float> vCamCalib2{fx, fy, cx, cy, K1, K2, K3, K4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpAtlas->AddCamera(mpCamera2);
                int leftLappingBegin = fSettings["Camera.lappingBegin"];
                int leftLappingEnd = fSettings["Camera.lappingEnd"];
                int rightLappingBegin = fSettings["Camera2.lappingBegin"];
                int rightLappingEnd = fSettings["Camera2.lappingEnd"];
                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;
                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;
                fSettings["Tlr"] >> mTlr;
                cout << "- mTlr: \n" << mTlr << endl;
                mpFrameDrawer->both = true;
            }
        }
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);
        const float k3 = fSettings["Camera.k3"];
        if(k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);
        mbf = fSettings["Camera.bf"];
        float fps = fSettings["Camera.fps"];
        if(fps == 0) {
            fps = 30;
        }
        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;
        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- bf: " << mbf << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        if(DistCoef.rows == 5) {
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        }
        cout << "- fps: " << fps << endl;
        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;
        if(mbRGB) {
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        } else {
            cout << "- color order: BGR (ignored if grayscale)" << endl;
        }
        // Load ORB parameters
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];
        // only ROI
        // 改进的特征点提取方法，分别对左右图像金字塔选择性提取
        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);//需要修改至执行函数
        if(sensor == System::STEREO || sensor == System::IMU_STEREO) {
            //二选一
//            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
            //改进的右图像特征点提取
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, true);
        }
        if(sensor == System::MONOCULAR || sensor == System::IMU_MONOCULAR) {
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
        }
        initID = 0;
        lastID = 0;
        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
        if(sensor == System::STEREO || sensor == System::RGBD || sensor == System::IMU_STEREO) {
            mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        if(sensor == System::RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if(fabs(mDepthMapFactor) < 1e-5) {
                mDepthMapFactor = 1;
            } else {
                mDepthMapFactor = 1.0f / mDepthMapFactor;
            }
        }
        if(sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO) {
            cv::Mat Tbc;
            fSettings["Tbc"] >> Tbc;
            cout << endl;
            cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;
            float freq, Ng, Na, Ngw, Naw;
            fSettings["IMU.Frequency"] >> freq;
            fSettings["IMU.NoiseGyro"] >> Ng;
            fSettings["IMU.NoiseAcc"] >> Na;
            fSettings["IMU.GyroWalk"] >> Ngw;
            fSettings["IMU.AccWalk"] >> Naw;
            const float sf = sqrt(freq);
            cout << endl;
            cout << "IMU frequency: " << freq << " Hz" << endl;
            cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
            cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
            cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
            cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;
            mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
            mnFramesToResetIMU = mMaxFrames;
        }
        mbInitWith3KFs = false;
        mnNumDataset = 0;
 
#ifdef SAVE_TIMES
        f_track_times.open("tracking_times.txt");
        f_track_times << "# ORB_Ext(ms), Stereo matching(ms), Preintegrate_IMU(ms), Pose pred(ms), LocalMap_track(ms), NewKF_dec(ms)" << endl;
        f_track_times << fixed ;
#endif
    }

    Tracking::~Tracking() {
        //f_track_stats.close();
#ifdef SAVE_TIMES
        f_track_times.close();
#endif
    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }

    void Tracking::SetStepByStep(bool bSet) {
        bStepByStep = bSet;
    }


    /**
     * @brief Tracking::GrabImageStereo 原抓取图片
     * @param imRectLeft
     * @param imRectRight
     * @param timestamp
     * @param filename
     * @return
     */
    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;
        mImRight = imRectRight;
        if(mImGray.channels() == 3) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if(mImGray.channels() == 4) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }
        if (mSensor == System::STEREO && !mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        } else if(mSensor == System::STEREO && mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
        } else if(mSensor == System::IMU_STEREO && !mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
        } else if(mSensor == System::IMU_STEREO && mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);
        }
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;
        Track();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
#ifdef SAVE_TIMES
        f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
        f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
        f_track_times << mTime_PreIntIMU << ",";
        f_track_times << mTime_PosePred << ",";
        f_track_times << mTime_LocalMapTrack << ",";
        f_track_times << mTime_NewKF_Dec << ",";
        f_track_times << t_track << endl;
#endif
       return mCurrentFrame.mTcw.clone();
    }

    /**
     * @brief Tracking::GrabImageStereo 抓取mask 图片
     * @param imRectLeft
     * @param imRectRight
     * @param imRectmaskleft
     * @param timestamp
     * @param filename
     * @return
     */
    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const cv::Mat &imRectmaskleft, const double &timestamp, string filename) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;
        mImRight = imRectRight;
        mImmaskleft = imRectmaskleft;
        //将mask改为单通道
        if(mImmaskleft.channels() == 3) {
            if(mbRGB) {
                cvtColor(mImmaskleft, mImmaskleft, CV_RGB2GRAY);
            } else {
                cvtColor(mImmaskleft, mImmaskleft, CV_BGR2GRAY);
            }
        } else if(mImmaskleft.channels() == 4) {
            if(mbRGB) {
                cvtColor(mImmaskleft, mImmaskleft, CV_RGBA2GRAY);
            } else {
                cvtColor(mImmaskleft, mImmaskleft, CV_BGRA2GRAY);
            }
        }

        if(mImGray.channels() == 3) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            }
        } else if(mImGray.channels() == 4) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            }
        }

        if(mImRight.channels() == 3) {
            if(mbRGB) {
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if(mImRight.channels() == 4) {
            if(mbRGB) {
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        /* 记录前后帧 */

        // copy to lastframe ..
        if(bMFlast) {
            MFLastFrame1 = Frame();
            MFLastFrame1 = Frame(MFLastFrame0);
            MFLastFrame1.premaskleft = MFLastFrame0.premaskleft;
            MFLastFrame1.maskleft = MFLastFrame0.maskleft;
            MFLastFrame1.imgLeft = MFLastFrame0.imgLeft;
            MFLastFrame1.imgRight = MFLastFrame0.imgRight;
            MFLastFrame1.mvKeysUndyna = MFLastFrame0.mvKeysUndyna;
            MFLastFrame1.mvKeysRightdyna = MFLastFrame0.mvKeysRightdyna;
            MFLastFrame1.mvKeysRightUndyna = MFLastFrame0.mvKeysRightUndyna;
            MFLastFrame1.Fcur = MFLastFrame0.Fcur;
            MFLastFrame1.refermvKeysdyna = MFLastFrame0.refermvKeysdyna;
            MFLastFrame1.maskObjDistance = MFLastFrame0.maskObjDistance;
            MFLastFrame1.maskclassnums = MFLastFrame0.maskclassnums;
            MFLastFrame1.goodmaskminpoint = MFLastFrame0.goodmaskminpoint;
        }
        if(bmcur) {
            MFLastFrame0 = Frame();
            MFLastFrame0 = Frame(mCurrentFrame);
            MFLastFrame0.premaskleft = mCurrentFrame.premaskleft;
            MFLastFrame0.maskleft = mCurrentFrame.maskleft;
            MFLastFrame0.imgLeft = mCurrentFrame.imgLeft;
            MFLastFrame0.imgRight = mCurrentFrame.imgRight;
            MFLastFrame0.mvKeysUndyna = mCurrentFrame.mvKeysUndyna;
            MFLastFrame0.mvKeysRightdyna = mCurrentFrame.mvKeysRightdyna;
            MFLastFrame0.mvKeysRightUndyna = mCurrentFrame.mvKeysRightUndyna;
            MFLastFrame0.Fcur = mCurrentFrame.Fcur;
            MFLastFrame0.refermvKeysdyna = mCurrentFrame.refermvKeysdyna;
            MFLastFrame0.maskObjDistance = mCurrentFrame.maskObjDistance;
            MFLastFrame0.maskclassnums = mCurrentFrame.maskclassnums;
            MFLastFrame0.goodmaskminpoint = mCurrentFrame.goodmaskminpoint;
            bMFlast = true;
        }


        /*  记录前后关键帧 */
        if(bmcur && (mCurrentFrame.mnId == mnLastKeyFrameId)) {
            // 上上关键帧记录保存到MFLast1KeyFrame
            if(bMFlast) {
                MFLast1KeyFrame = Frame(MFLast0KeyFrame);
                MFLast1KeyFrame.maskleft = MFLast0KeyFrame.maskleft;
                MFLast1KeyFrame.imgLeft = MFLast0KeyFrame.imgLeft;
                MFLast1KeyFrame.imgRight = MFLast0KeyFrame.imgRight;
            }
            // 当前关键帧记录保存到MFLast0KeyFrame
            MFLast0KeyFrame = Frame(mCurrentFrame);
            MFLast0KeyFrame.maskleft = mCurrentFrame.maskleft;
            MFLast0KeyFrame.imgLeft = mCurrentFrame.imgLeft;
            bMFlast = true;
        }

          if (mSensor == System::STEREO && !mpCamera2) {
              //　with mask
            mCurrentFrame = Frame(mImGray, imGrayRight, mImmaskleft, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
            mCurrentFrame.imgLeft = mImGray;
            mCurrentFrame.imgRight = imGrayRight;
            mCurrentFrame.premaskleft = mImmaskleft;
            //膨胀掩码,目的是包容mask边界点
            mImmaskleft = dilatemask(mImmaskleft);
            mCurrentFrame.maskleft = mImmaskleft;
            bmcur = true;
        } else if(mSensor == System::STEREO && mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
        } else if(mSensor == System::IMU_STEREO && !mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
        } else if(mSensor == System::IMU_STEREO && mpCamera2) {
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);
        }
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;


        /* 前后帧显示匹配显示 */


        //显示前后帧匹配
        Mat frontandbackimg;
        if(mCurrentFrame.mnId - mnLastRelocFrameId > 4) {
            // 测试前后帧匹配函数
            Frontandbackframematching();
            Combineimg(MFLastFrame0.imgLeft, mCurrentFrame.imgLeft, frontandbackimg);
        }
        //　跟踪
        Track();
        //计算前后帧Ｅ和Ｆ矩阵（不是前后关键帧）
        if(mCurrentFrame.mnId - mnLastRelocFrameId > 4) {
//            CalculMatEF();//这个方法有点蠢，其实都已经写好了，在ComputeF12函数中，大意了，测试输出相同
            // 调用作者方法计算
            if(!MFLastFrame0.mTcw.empty() && !mCurrentFrame.mTcw.empty()){
                F = ComputeframeF12(&MFLastFrame0, &mCurrentFrame);
            }
        }
        // trracking time
        // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
        
#ifdef SAVE_TIMES
        f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
        f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
        f_track_times << mTime_PreIntIMU << ",";
        f_track_times << mTime_PosePred << ",";
        f_track_times << mTime_LocalMapTrack << ",";
        f_track_times << mTime_NewKF_Dec << ",";
        f_track_times << t_track << endl;
#endif
        return mCurrentFrame.mTcw.clone();
    }

    // 膨胀mask
    cv::Mat Tracking::dilatemask(cv::Mat &mask) {
        cv::Mat MaskLeft_dil = mask.clone();
        int dilation_size = 4;
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size( 2 * dilation_size + 1, 2 * dilation_size + 1 ),
                                               cv::Point( dilation_size, dilation_size ) );
        dilate(mask, MaskLeft_dil, kernel);
        return MaskLeft_dil;
    }

    // 计算本质矩阵E和基础矩阵F

    cv::Mat Tracking::CalculMatEF() {

        cv::Mat tT, R;
        // tT 这里求反对称矩阵其实是乘-1的
        tT = (cv::Mat_<float>(3, 3) << 0, mVelocity.at<float>(2, 3), -mVelocity.at<float>(1, 3),
              -mVelocity.at<float>(2, 3), 0, mVelocity.at<float>(0, 3),
              mVelocity.at<float>(1, 3),  -mVelocity.at<float>(0, 3), 0);
        R = mVelocity.rowRange(0, 3).colRange(0, 3); //赋值

        mCurrentFrame.Ecur = tT * R.t();//注意R.t()为什么
        // 公式slam十四讲167页
        Mat Kinv, KTtemp;
        Kinv = mK.inv();
        KTtemp = Kinv.t();
        mCurrentFrame.Fcur = KTtemp * mCurrentFrame.Ecur * Kinv;
        std::cout << "自己计算矩阵Ｆ" << std::endl;
        printMat(mCurrentFrame.Fcur);

    }


    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename) {
        mImGray = imRGB;
        cv::Mat imDepth = imD;
        if(mImGray.channels() == 3) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            }
        } else if(mImGray.channels() == 4) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            }
        }
        if((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F) {
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
        }
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;
        Track();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
#ifdef SAVE_TIMES
        f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
        f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
        f_track_times << mTime_PreIntIMU << ",";
        f_track_times << mTime_PosePred << ",";
        f_track_times << mTime_LocalMapTrack << ",";
        f_track_times << mTime_NewKF_Dec << ",";
        f_track_times << t_track << endl;
#endif
        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename) {
        mImGray = im;
        if(mImGray.channels() == 3) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            }
        } else if(mImGray.channels() == 4) {
            if(mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            }
        }
        if (mSensor == System::MONOCULAR) {
            if(mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames) {
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            } else {
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            }
        } else if(mSensor == System::IMU_MONOCULAR) {
            if(mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
                cout << "init extractor" << endl;
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            } else {
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            }
        }
        if (mState == NO_IMAGES_YET) {
            t0 = timestamp;
        }
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;
        lastID = mCurrentFrame.mnId;
        Track();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();

#ifdef SAVE_TIMES
        f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
        f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
        f_track_times << mTime_PreIntIMU << ",";
        f_track_times << mTime_PosePred << ",";
        f_track_times << mTime_LocalMapTrack << ",";
        f_track_times << mTime_NewKF_Dec << ",";
        f_track_times << t_track << endl;
#endif
        return mCurrentFrame.mTcw.clone();
    }


    void Tracking::GrabImuData(const IMU::Point &imuMeasurement) {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }

    void Tracking::PreintegrateIMU() {
        cout << "start preintegration" << endl;
        if(!mCurrentFrame.mpPrevFrame) {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }
        cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;
        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mlQueueImuData.size());
        if(mlQueueImuData.size() == 0) {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }
        while(true) {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if(!mlQueueImuData.empty()) {
                    IMU::Point *m = &mlQueueImuData.front();
                    cout.precision(17);
                    if(m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.001l) {
                        mlQueueImuData.pop_front();
                    } else if(m->t < mCurrentFrame.mTimeStamp - 0.001l) {
                        mvImuFromLastFrame.push_back(*m);
                        mlQueueImuData.pop_front();
                    } else {
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                } else {
                    break;
                    bSleep = true;
                }
            }
            if(bSleep) {
                usleep(500);
            }
        }
        const int n = mvImuFromLastFrame.size() - 1;
        IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);
        for(int i = 0; i < n; i++) {
            float tstep;
            cv::Point3f acc, angVel;
            if((i == 0) && (i < (n - 1))) {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            } else if(i < (n - 1)) {
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            } else if((i > 0) && (i == (n - 1))) {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
                tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
            } else if((i == 0) && (i == (n - 1))) {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }
            if (!mpImuPreintegratedFromLastKF) {
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            }
            mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
        }
        mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;
        if(!mpLastKeyFrame) {
            cout << "last KF is empty!" << endl;
        }
        mCurrentFrame.setIntegrated();
        Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }


    bool Tracking::PredictStateIMU() {
        if(!mCurrentFrame.mpPrevFrame) {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }
        if(mbMapUpdated && mpLastKeyFrame) {
            const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
            const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
            const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();
            const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mpImuPreintegratedFromLastKF->dT;
            cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
            cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
            cv::Mat Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        } else if(!mbMapUpdated) {
            const cv::Mat twb1 = mLastFrame.GetImuPosition();
            const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
            const cv::Mat Vwb1 = mLastFrame.mVw;
            const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;
            cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
            cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
            cv::Mat Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);
            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias = mLastFrame.mImuBias;
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        } else {
            cout << "not IMU prediction!!" << endl;
        }
        return false;
    }


    void Tracking::ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx,  float &bwy, float &bwz) {
        const int N = vpFs.size();
        vector<float> vbx;
        vbx.reserve(N);
        vector<float> vby;
        vby.reserve(N);
        vector<float> vbz;
        vbz.reserve(N);
        cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat grad  = cv::Mat::zeros(3, 1, CV_32F);
        for(int i = 1; i < N; i++) {
            Frame *pF2 = vpFs[i];
            Frame *pF1 = vpFs[i - 1];
            cv::Mat VisionR = pF1->GetImuRotation().t() * pF2->GetImuRotation();
            cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
            cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t() * VisionR;
            cv::Mat e = IMU::LogSO3(E);
            assert(fabs(pF2->mTimeStamp - pF1->mTimeStamp - pF2->mpImuPreintegratedFrame->dT) < 0.01);
            cv::Mat J = -IMU::InverseRightJacobianSO3(e) * E.t() * JRg;
            grad += J.t() * e;
            H += J.t() * J;
        }
        cv::Mat bg = -H.inv(cv::DECOMP_SVD) * grad;
        bwx = bg.at<float>(0);
        bwy = bg.at<float>(1);
        bwz = bg.at<float>(2);
        for(int i = 1; i < N; i++) {
            Frame *pF = vpFs[i];
            pF->mImuBias.bwx = bwx;
            pF->mImuBias.bwy = bwy;
            pF->mImuBias.bwz = bwz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            pF->mpImuPreintegratedFrame->Reintegrate();
        }
    }

    void Tracking::ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax,  float &bay, float &baz) {
        const int N = vpFs.size();
        const int nVar = 3 * N + 3; // 3 velocities/frame + acc bias
        const int nEqs = 6 * (N - 1);
        cv::Mat J(nEqs, nVar, CV_32F, cv::Scalar(0));
        cv::Mat e(nEqs, 1, CV_32F, cv::Scalar(0));
        cv::Mat g = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
        for(int i = 0; i < N - 1; i++) {
            Frame *pF2 = vpFs[i + 1];
            Frame *pF1 = vpFs[i];
            cv::Mat twb1 = pF1->GetImuPosition();
            cv::Mat twb2 = pF2->GetImuPosition();
            cv::Mat Rwb1 = pF1->GetImuRotation();
            cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
            cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
            cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
            cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
            float t12 = pF2->mpImuPreintegratedFrame->dT;
            // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
            J.rowRange(6 * i, 6 * i + 3).colRange(3 * i, 3 * i + 3) += cv::Mat::eye(3, 3, CV_32F) * t12;
            J.rowRange(6 * i, 6 * i + 3).colRange(3 * N, 3 * N + 3) += Rwb1 * JP12;
            e.rowRange(6 * i, 6 * i + 3) = twb2 - twb1 - 0.5f * g * t12 * t12 - Rwb1 * dP12;
            // Velocity v2=v1+g*t+R1*dV12
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * i, 3 * i + 3) += -cv::Mat::eye(3, 3, CV_32F);
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * (i + 1), 3 * (i + 1) + 3) += cv::Mat::eye(3, 3, CV_32F);
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * N, 3 * N + 3) -= Rwb1 * JV12;
            e.rowRange(6 * i + 3, 6 * i + 6) = g * t12 + Rwb1 * dV12;
        }
        cv::Mat H = J.t() * J;
        cv::Mat B = J.t() * e;
        cv::Mat x(nVar, 1, CV_32F);
        cv::solve(H, B, x);
        bax = x.at<float>(3 * N);
        bay = x.at<float>(3 * N + 1);
        baz = x.at<float>(3 * N + 2);
        for(int i = 0; i < N; i++) {
            Frame *pF = vpFs[i];
            x.rowRange(3 * i, 3 * i + 3).copyTo(pF->mVw);
            if(i > 0) {
                pF->mImuBias.bax = bax;
                pF->mImuBias.bay = bay;
                pF->mImuBias.baz = baz;
                pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            }
        }
    }

    void Tracking::ResetFrameIMU() {
        // TODO To implement...
    }


    void Tracking::Track() {
#ifdef SAVE_TIMES
        mTime_PreIntIMU = 0;
        mTime_PosePred = 0;
        mTime_LocalMapTrack = 0;
        mTime_NewKF_Dec = 0;
#endif
        if (bStepByStep) {
            while(!mbStep) {
                usleep(500);
            }
            mbStep = false;
        }
        if(mpLocalMapper->mbBadImu) {
            cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
            mpSystem->ResetActiveMap();
            return;
        }
        Map *pCurrentMap = mpAtlas->GetCurrentMap();
        if(mState != NO_IMAGES_YET) {
            if(mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                CreateMapInAtlas();
                return;
            } else if(mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 3.0) {
                cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
                if(mpAtlas->isInertial()) {
                    if(mpAtlas->isImuInitialized()) {
                        cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                        if(!pCurrentMap->GetIniertialBA2()) {
                            mpSystem->ResetActiveMap();
                        } else {
                            CreateMapInAtlas();
                        }
                    } else {
                        cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                        mpSystem->ResetActiveMap();
                    }
                }
                return;
            }
        }
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame) {
            mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
        }
        if(mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }
        mLastProcessedState = mState;
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap) {
#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#endif
            PreintegrateIMU();
#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            mTime_PreIntIMU = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
#endif
        }
        mbCreatedMap = false;
        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);
        mbMapUpdated = false;
        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if(nCurMapChangeIndex > nMapChangeIndex) {
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }
        if(mState == NOT_INITIALIZED) {
            if(mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO) {
                StereoInitialization();
            } else {
                MonocularInitialization();
            }
            //更新帧绘制器中存储的最新状态
            mpFrameDrawer->Update(this);
            //这个状态量在上面的初始化函数中被更新
            if(mState != OK) { // If rightly initialized, mState=OK
                mLastFrame = Frame(mCurrentFrame);
                return;
            }
            if(mpAtlas->GetAllMaps().size() == 1) {
                mnFirstFrameId = mCurrentFrame.mnId;
            }
        } else {
            // System is initialized. Track Frame.
            bool bOK;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
            // tracking 类构造时默认为false。在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
            if(!mbOnlyTracking) {
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point timeStartPosePredict = std::chrono::steady_clock::now();
#endif
                // State OK
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                // Step 2：跟踪进入正常SLAM模式，有地图更新
                // 正常初始化成功
                if(mState == OK) {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    // Step 2.1 检查并更新上一帧被替换的MapPoints
                    // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                    CheckReplacedInLastFrame();
                    // Step 2.2 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                    // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
                    // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
                    // mnLastRelocFrameId 上一次重定位的那一帧
                    if((mVelocity.empty() && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        //Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                        // 用最近的关键帧来跟踪当前的普通帧
                        // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                        // 优化每个特征点都对应3D点重投影误差即可得到位姿
                        bOK = TrackReferenceKeyFrame();
                    } else {
                        //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                        // 用最近的普通帧来跟踪当前的普通帧
                        // 根据恒速模型设定当前帧的初始位姿
                        // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                        // 优化每个特征点所对应3D点的投影误差即可得到位姿
                        bOK = TrackWithMotionModel();
                        if(!bOK)
                            //根据恒速模型失败了，只能根据参考关键帧来跟踪
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                    if (!bOK) {
                        if ( mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                                (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
                            mState = LOST;
                        } else if(pCurrentMap->KeyFramesInMap() > 10) {
                            cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                        } else {
                            mState = LOST;
                        }
                    }
                } else {
                    if (mState == RECENTLY_LOST) {
                        Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);
                        bOK = true;
                        if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
                            if(pCurrentMap->isImuInitialized()) {
                                PredictStateIMU();
                            } else {
                                bOK = false;
                            }
                            if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost) {
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        } else {
                            // TODO fix relocalization
                            // 如果跟踪状态不成功,那么就只能重定位了
                            // BOW搜索，PnP求解位姿
                            bOK = Relocalization();
                            if(!bOK) {
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        }
                    } else if (mState == LOST) {
                        Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);
                        if (pCurrentMap->KeyFramesInMap() < 10) {
                            mpSystem->ResetActiveMap();
                            cout << "Reseting current map..." << endl;
                        } else {
                            CreateMapInAtlas();
                        }
                        if(mpLastKeyFrame) {
                            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
                        }
                        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
                        return;
                    }
                }
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point timeEndPosePredict = std::chrono::steady_clock::now();
                mTime_PosePred = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(timeEndPosePredict - timeStartPosePredict).count();
#endif
            } else {
                // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
                if(mState == LOST) {
                    if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
                        Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                    }
                    bOK = Relocalization();
                } else {
                    if(!mbVO) {
                        // In last frame we tracked enough MapPoints in the map
                        if(!mVelocity.empty()) {
                            bOK = TrackWithMotionModel();
                        } else {
                            bOK = TrackReferenceKeyFrame();
                        }
                    } else {
                        // In last frame we tracked mainly "visual odometry" points.
                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.
                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if(!mVelocity.empty()) {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization();
                        if(bOKMM && !bOKReloc) {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;
                            if(mbVO) {
                                for(int i = 0; i < mCurrentFrame.N; i++) {
                                    if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        } else if(bOKReloc) {
                            mbVO = false;
                        }
                        bOK = bOKReloc || bOKMM;
                    }
                }
            }
            if(!mCurrentFrame.mpReferenceKF) {
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            }
            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(!mbOnlyTracking) {
                if(bOK) {
#ifdef SAVE_TIMES
                    std::chrono::steady_clock::time_point time_StartTrackLocalMap = std::chrono::steady_clock::now();
#endif
                    bOK = TrackLocalMap();
#ifdef SAVE_TIMES
                    std::chrono::steady_clock::time_point time_EndTrackLocalMap = std::chrono::steady_clock::now();
                    mTime_LocalMapTrack = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndTrackLocalMap - time_StartTrackLocalMap).count();
#endif
                }
                if(!bOK) {
                    cout << "Fail to track local map!" << endl;
                }
            } else {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if(bOK && !mbVO) {
                    bOK = TrackLocalMap();
                }
            }
            if(bOK) {
                mState = OK;
            } else if (mState == OK) {
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
                    Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                    if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) {
                        cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                        mpSystem->ResetActiveMap();
                    }
                    mState = RECENTLY_LOST;
                } else {
                    mState = LOST;    // visual to lost
                }
                if(mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames) {
                    mTimeStampLost = mCurrentFrame.mTimeStamp;
                }
            }
            // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
            if((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) && ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && pCurrentMap->isImuInitialized()) {
                // TODO check this situation
                Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
                Frame *pF = new Frame(mCurrentFrame);
                pF->mpPrevFrame = new Frame(mLastFrame);
                // Load preintegration
                pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
            }
            if(pCurrentMap->isImuInitialized()) {
                if(bOK) {
                    if(mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU)) {
                        cout << "RESETING FRAME!!!" << endl;
                        ResetFrameIMU();
                    } else if(mCurrentFrame.mnId > (mnLastRelocFrameId + 30)) {
                        mLastBias = mCurrentFrame.mImuBias;
                    }
                }
            }
            // Update drawer
            mpFrameDrawer->Update(this);
            if(!mCurrentFrame.mTcw.empty()) {
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            }
            if(bOK || mState == RECENTLY_LOST) {
                // Update motion model
                if(!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else {
                    mVelocity = cv::Mat();
                }
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
                    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                }
                // Clean VO matches
                for(int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if(pMP)
                        if(pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                // Delete temporal MapPoints
                for(list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point timeStartNewKF = std::chrono::steady_clock::now();
#endif
                bool bNeedKF = NeedNewKeyFrame();
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point timeEndNewKF = std::chrono::steady_clock::now();
                mTime_NewKF_Dec = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(timeEndNewKF - timeStartNewKF).count();
#endif
                // Check if we need to insert a new keyframe
                if(bNeedKF && (bOK || (mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)))) {
                    CreateNewKeyFrame();
                }
                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for(int i = 0; i < mCurrentFrame.N; i++) {
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                }
            }
            // Reset if the camera get lost soon after initialization
            if(mState == LOST) {
                if(pCurrentMap->KeyFramesInMap() <= 5) {
                    mpSystem->ResetActiveMap();
                    return;
                }
                if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
                    if (!pCurrentMap->isImuInitialized()) {
                        Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                        mpSystem->ResetActiveMap();
                        return;
                    }
                CreateMapInAtlas();
            }
            if(!mCurrentFrame.mpReferenceKF) {
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            }
            mLastFrame = Frame(mCurrentFrame);
        }
        if(mState == OK || mState == RECENTLY_LOST) {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if(!mCurrentFrame.mTcw.empty()) {
                cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState == LOST);
            } else {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState == LOST);
            }
        }
    }//end of tracking


    void Tracking::StereoInitialization() {
        if(mCurrentFrame.N > 500) {
            if (mSensor == System::IMU_STEREO) {
                if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated) {
                    cout << "not IMU meas" << endl;
                    return;
                }
                if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA) < 0.5) {
                    cout << cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA) << endl;
                    cout << "not enough acceleration" << endl;
                    return;
                }
                if(mpImuPreintegratedFromLastKF) {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }
            // Set Frame pose to the origin (In case of inertial SLAM to imu)
            if (mSensor == System::IMU_STEREO) {
                cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3).clone();
                cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).col(3).clone();
                mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3, 1, CV_32F));
            } else {
                mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            }
            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
            // Insert KeyFrame in the map
            mpAtlas->AddKeyFrame(pKFini);
            // Create MapPoints and asscoiate to KeyFrame
            if(!mpCamera2) {
                cerr << "Use the first initialization method like slam2！　" << endl;
                for(int i = 0; i < mCurrentFrame.N; i++) {
                    float z = mCurrentFrame.mvDepth[i];
                    if(z > 0) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);//左相机
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKFini, i); //a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();// b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
                        pNewMP->UpdateNormalAndDepth(); // c.更新该MapPoint平均观测方向以及观测距离的范围
                        mpAtlas->AddMapPoint(pNewMP);// 在地图中添加该MapPoint
                        mCurrentFrame.mvpMapPoints[i] = pNewMP; // 表示该KeyFrame的哪个特征点可以观测到哪个3D点
                    }
                }
            } else {
                cerr << "Use a second new initialization method！" << endl;
                for(int i = 0; i < mCurrentFrame.Nleft; i++) {
                    int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                    if(rightIndex != -1) {
                        cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKFini, i);
                        pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);
                        pKFini->AddMapPoint(pNewMP, i);
                        pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);
                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                    }
                }
            }
            Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
            mpLocalMapper->InsertKeyFrame(pKFini);
            // 更新当前帧为上一帧
            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;
            mnLastRelocFrameId = mCurrentFrame.mnId;
            mvpLocalKeyFrames.push_back(pKFini);
            // 我现在的想法是，这个点只是暂时被保存在了 Tracking 线程之中， 所以称之为 local
            // 初始化之后，通过双目图像生成的地图点，都应该被认为是局部地图点
            mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;
            // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
            // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
            mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
            mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            //追踪成功
            mState = OK;
        }
    }

    void Tracking::Frontandbackframematching() {
        //　测试是不是因为点以改变？
        // 跟踪初始化时前两帧之间的匹配
        // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
        std::vector<cv::Point2f> MFlastpoints0, MFcurrentpoints;
        MFlastpoints0.resize(MFLastFrame0.mvKeysUndyna.size());
        //MFcurrent 记录"上一帧"所有特征点
        MFcurrentpoints.resize(mCurrentFrame.mvKeysUndyna.size());
        //　取出前一帧点
        for(size_t i = 0; i < MFLastFrame0.mvKeysUndyna.size(); i++) {
            MFlastpoints0[i] = MFLastFrame0.mvKeysUndyna[i].pt;
        }
        //　取出当前帧的点
        for(size_t i = 0; i < mCurrentFrame.mvKeysUndyna.size(); i++) {
            MFcurrentpoints[i] = mCurrentFrame.mvKeysUndyna[i].pt;
        }
        //在mInitialFrame与mCurrentFrame中找匹配的特征点对
        ORBmatcher MFmatcher(
            0.7,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
            true);      //检查特征点的方向
        MFmatcher.MFSearchForSegmentregiondyna(MFLastFrame0, mCurrentFrame, MFlastpoints0, FrontbackMatches, FrontbackoutMatches, 100);

        // 先计算前一帧
        Mat T12 = ComputeframeT12(&MFLastFrame0, &MFLastFrame1);//当前帧到前一帧变换
        Mat T21 = T12.inv();
        Mat R12 = T12.rowRange(0, 3).colRange(0, 3).clone();
        Mat t12 = T12.rowRange(0, 3).col(3).clone();

        // 以第一个相机的光心作为世界坐标系, 定义相机的投影矩阵
        cv::Mat Plast(3, 4,				//矩阵的大小是3x4
                      CV_32F,			//数据类型是浮点数
                      cv::Scalar(0));	//初始的数值是0
        cv::Mat Pcur21(3, 4,				//矩阵的大小是3x4
                       CV_32F,			//数据类型是浮点数
                       cv::Scalar(0));	//初始的数值是0

        //将整个K矩阵拷贝到P1矩阵的左侧3x3矩阵，因为 K*I = K
        MFLastFrame0.mK.copyTo(Plast.rowRange(0, 3).colRange(0, 3));
        MFLastFrame1.mK.copyTo(Pcur21.rowRange(0, 3).colRange(0, 3));

        // Camera 2 Projection Matrix K[R|t]
        // 计算第二个相机的投影矩阵 P2=K*[R|t]
        cv::Mat Pcur(3, 4, CV_32F);
        T12.rowRange(0, 3).colRange(0, 4).copyTo(Pcur.rowRange(0, 3).colRange(0, 4)); //赋值给P2

        cv::Mat Plast21(3, 4, CV_32F);
        T21.rowRange(0, 3).colRange(0, 4).copyTo(Plast21.rowRange(0, 3).colRange(0, 4)); //赋值给P2

        for(size_t i = 0, iend = MFLastFrame0.mvKeys.size(); i < iend; i++) {
            //是否和当前帧匹配
            if(FrontbackMatches[i] == -1) {
                continue;
            }
            //前一帧
            const cv::KeyPoint &kplast = MFLastFrame0.mvKeys[i];
            const cv::KeyPoint &kpcur = mCurrentFrame.mvKeys[FrontbackMatches[i]];

            //测试
            Mat pl = (cv::Mat_<float>(3, 1, CV_32F) << MFLastFrame0.mvKeys[i].pt.x, MFLastFrame0.mvKeys[i].pt.y, 1);
            Mat pc = (cv::Mat_<float>(3, 1, CV_32F) << mCurrentFrame.mvKeys[FrontbackMatches[i]].pt.x, mCurrentFrame.mvKeys[FrontbackMatches[i]].pt.y, 1);
            Mat pl_n = MFLastFrame0.mK.inv() * pl;//投影
            Mat pc_n = mCurrentFrame.mK.inv() * pc;
            pl_n = pl_n.rowRange(0, 3) / pl_n.at<float>(2);//归一化
            pc_n = pc_n.rowRange(0, 3) / pc_n.at<float>(2);//归一化

            cv::Mat p3dlast = pl_n;
            cv::Mat p3dcur21 = pc_n;
            if(mCurrentFrame.mvDepth[FrontbackMatches[i]] != -1) {
                //前一帧
                float Trz = (Lcx + (p3dcur21.at<float>(0) / p3dcur21.at<float>(2)) * fx) / (p3dcur21.at<float>(0) / p3dcur21.at<float>(2));

                // 求解
                // 方程 https://zhuanlan.zhihu.com/p/112592149
                // f_ref.normalize();//参考帧上的点在相机坐标系下的归一化的坐标x_1
                // f_curr.normalize();//当前帧上的点在相机坐标系下的归一化的坐标x_2
                //    d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC         // d_ref=x_1,对应s_1x_1=s_2(R*x_2)+t   s_1 x_1=s_2(R*x_2)+t
                // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t] //  s_1 x_1^T x_1=s_2 x_1^T x_2+x_1^T t
                //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t] //  s_2 x_2^T x_1=s_2 x_2^T x_2+x_2^T t
                // 二阶方程用克莱默法则求解并解之
                // 乘转置
                Mat denominator = -p3dlast.t() * p3dlast * p3dcur21.t() * R12.inv() * p3dcur21 + p3dlast.t() * R12.inv() * p3dcur21 * p3dcur21.t() * p3dlast;
                Mat molecule_last = p3dlast.t() * t12 * p3dcur21.t() * R12.inv() * p3dcur21 - p3dlast.t() * R12.inv() * p3dcur21 * p3dcur21.t() * t12;
                Mat molecule_cur = -p3dlast.t() * p3dlast * p3dcur21.t() * t12 + p3dlast.t() * t12 * p3dcur21.t() * p3dlast;

                Mat d_last = molecule_last / denominator;
                Mat d_cur = molecule_cur / denominator;
            }
        }

        /* DynaSLAM */
// 测试匹配的点对
        int outmaskdistance = 0, tempmasknumforfb = 0;
        vector<float> statispointdistance;//静态地图点距离
        float statisdistance = 0;
        float maskdistance = 0;
        vector<vector<int> > statispointId(mCurrentFrame.maskclassnums);//静态点Ｉｄ,根据mask 区域分类，存储对应mask列区域的特征点

        vector<vector<float> > maskpointdistance(mCurrentFrame.maskclassnums);//mask对象距离
        vector<vector<int> > maskpointId(mCurrentFrame.maskclassnums);//记录要去除前后帧匹配上的点，但是实际是需要去去除当前帧msak的所有点
        vector<bool> dynamaskerea(mCurrentFrame.maskclassnums, false); //记录这个mask为动态否

        // 极限距离判断是否为动态
        vector<float> statispointEpipolardistance;//静态地图点极线距离
        vector<vector<float> > maskpointEpipolardistance(mCurrentFrame.maskclassnums);//mask对象距离
        //　测试边界
        vector<vector<int> > maskbord(mCurrentFrame.maskclassnums, vector<int>());//储存mask边界
        Calculmaskborder(mCurrentFrame.maskleft, mCurrentFrame.maskclassnums, maskbord);

        // 保存位于mask这一列的点
        vector<vector<Point2f> > maskareastatispoint(mCurrentFrame.maskclassnums, vector<Point2f>());//储存mask 列的点
        vector<vector<float> > maskareastatispointEpipolaraverdistance(mCurrentFrame.maskclassnums, vector<float>());//储存mask这些静态点的极线距离
        vector<vector<float> > maskareastatispointOpticalflowaverdistance(mCurrentFrame.maskclassnums, vector<float>());// 储存mask这些静态点的光流距离

        // 记录当前帧curMasknum和上一帧frontMasknum所有点的对应关系
        vector<vector<int> > curMasknum2frontMasknums(mCurrentFrame.maskclassnums, vector<int>());
        // 初始化一个记录当前帧和前一帧mask匹配对应，即跟踪的对应关系
        vector<vector<int> > maskNumsmatch;

        Mat currF = ORB_SLAM3::F.clone();

        for(int i = 0, fbnum = FrontbackMatches.size(); i < fbnum; i++) {
            if(FrontbackMatches[i] == -1) {
                continue;
            } else {
                if(((int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.y),
                        round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.x)) == 0) || ((int)mCurrentFrame.maskleft.at<unsigned char>
                                (round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.y), round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.x)) == 255)) {
                   float Epipolardistance = CalculEpipolardistance(MFLastFrame0.mvKeysUndyna[i].pt, mCurrentFrame.mvKeysUndyna[FrontbackMatches[i]].pt, currF);
                   statispointEpipolardistance.emplace_back(Epipolardistance);//极线距离
                    float opticaldistance = Calcultwopointdistance(MFLastFrame0.mvKeysUndyna[i].pt, mCurrentFrame.mvKeysUndyna[FrontbackMatches[i]].pt);
                    statispointdistance.emplace_back(opticaldistance);//光流距离

                    //使用区域计算点极限距离方法
                    for(int pointboder = 0; pointboder < mCurrentFrame.maskclassnums; pointboder++) {
                        //step1 判断在那个列区域类  //pt.x对应的列
                        if(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.x > maskbord[pointboder][0] &&
                                mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.x < maskbord[pointboder][1]) {
                            maskareastatispoint[pointboder].push_back(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt); //将对应列区域的点保存在对应的mask区域
                            // 将这两个静态点的极限距离保存到　maskareastatispointEpipolaraverdistance
                            maskareastatispointEpipolaraverdistance[pointboder].push_back(Epipolardistance);
                            // 光流距离
                            maskareastatispointOpticalflowaverdistance[pointboder].push_back(opticaldistance);
                        }

                    }

                } else {

                    // j为当前帧的mask 序号
                    int curMasknum = (250 - (int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.y), round(mCurrentFrame.refermvKeysdyna[FrontbackMatches[i]].pt.x))) / 5;
                    if(curMasknum < 0 || curMasknum >= mCurrentFrame.maskclassnums) {
                        continue;
                    }
                 float Epipolardistance = CalculEpipolardistance(MFLastFrame0.mvKeysUndyna[i].pt, mCurrentFrame.mvKeysUndyna[FrontbackMatches[i]].pt, currF);
                    maskpointEpipolardistance[curMasknum].emplace_back(Epipolardistance);//极线距离

                    maskdistance = Calcultwopointdistance(MFLastFrame0.mvKeysUndyna[i].pt, mCurrentFrame.mvKeysUndyna[FrontbackMatches[i]].pt);//光流距离
                    maskpointdistance[curMasknum].emplace_back(maskdistance);
                    maskpointId[curMasknum].emplace_back(FrontbackMatches[i]);
                    // 距离当前帧序号ｊ的掩码值对应的上一帧的掩码值，注意有匹配的，或者没有匹配的情况
                    int frontMasknum = (250 - (int)MFLastFrame0.maskleft.at<unsigned char>(round(MFLastFrame0.refermvKeysdyna[i].pt.y), round(MFLastFrame0.refermvKeysdyna[i].pt.x))) / 5;
                    if (frontMasknum >= 0 && frontMasknum < MFLastFrame0.maskclassnums) { //排除匹配到背景点的情况
                        // 统计当前帧curMasknum和前一帧frontMasknum的对饮关系
                        curMasknum2frontMasknums[curMasknum].push_back(frontMasknum);//这样就将当前帧对应的msak一上一帧对应的mask 关联，这里是考虑到无匹配到其他mask对对象上，故后续筛选一一对应关系
                    }
                }
            }
        }//end for points

        // 对curMasknum2frontMasknums选出一一对应的关系，注意这里的curMasknum2frontMasknums的当前mask存在上一次没有mask的情况
        ChoicecurMaskOnebyOnefrontMask(curMasknum2frontMasknums, maskNumsmatch);

        // 计算当前帧组合中运动对象的速度
        CurrentSpeed(mCurrentFrame, MFLastFrame0, maskNumsmatch);

        // 记录运动的记录
        float moveZdistance;
        if(mVelocity.empty()) {
            moveZdistance = 0.44;
        } else {
            moveZdistance = -mVelocity.at<float>(2, 3); //注意已经取取反
        }

        float statispiontsaverdistance;
        if(statispointdistance.size() == 0) {
            cerr << "没有静态匹配点" << endl;
            statispiontsaverdistance = INT_MIN;
        } else {
            statispiontsaverdistance = Calcuaverdistance(statispointdistance);
        }

        //开始提出动态对象点
        // step 1
        vector<float> maskaverdistance[mCurrentFrame.maskclassnums];
        if(statispiontsaverdistance != INT_MIN) {
            for(int maskclass = 0; maskclass < mCurrentFrame.maskclassnums; maskclass++) {
                if(maskpointdistance[maskclass].size() == 0) {
                    maskaverdistance[maskclass].push_back(statispiontsaverdistance);
                } else {
                    maskaverdistance[maskclass].push_back(Calcuaverdistance(maskpointdistance[maskclass]));
                }
            }
        }

        float th0 = 0.7;
        float th = 0.9; //分割极线距离的阈值 statisEpipolarpointsdistance < ChoiceoneEpipolardistance(maskpointEpipolardistance[dynamask], th)
        float th1 = 0.5; //分割极线距离的阈值 CalcullowerThaverdis(maskareastatispointEpipolaraverdistance[dynamask], th1
        float th2 = 0.6; //分割极线距离的阈值 CalcullowerThaverdis(maskpointEpipolardistance[dynamask], th2
        float th3 = 0.6; //越大，判别的运动距离越大 abs(maskaverdistance[dynamask].front() - statispiontsaverdistance) / statispiontsaverdistance > th3
        float th4 = 0.5;
        // 仅仅只是选取th阈值出的点
        float statisEpipolarpointsdistance = ChoiceoneEpipolardistance(statispointEpipolardistance, th); //计算静态点的极线距离
        // 选取低于th 点的平均点
//        float statisEpipolarpointsAverdistance = CalcullowerThaverdis(statispointEpipolardistance, th);


        // 去除动态点对象，记录那些点应该去除
        vector<int> willremovepoint, willremovepointRight;
        for(int dynamask = 0; dynamask < mCurrentFrame.maskclassnums; dynamask++) {
            if(statispointdistance.size() == 0) {
                break;
            }
            if(maskaverdistance[dynamask].front() == statispiontsaverdistance) {
                continue;
            }
            if(
                ((CalcullowerThaverdis(maskareastatispointEpipolaraverdistance[dynamask], th1) <  CalcullowerThaverdis(maskpointEpipolardistance[dynamask], th2)
                ) && WhetherDynawithDistance(maskNumsmatch, dynamask, mCurrentFrame.maskObjDistance, MFLastFrame0.maskObjDistance, maskbord, moveZdistance)
                || abs(maskaverdistance[dynamask].front() - CalcullowerThaverdis(maskareastatispointOpticalflowaverdistance[dynamask], th0)) /
                    abs(CalcullowerThaverdis(maskareastatispointOpticalflowaverdistance[dynamask], th0)) > th3)
            ) {
            // 设置去除动态点的mask
                dynamaskerea[dynamask] = true;
                for(int point = 0; point < maskpointId[dynamask].size(); point++) {
                    willremovepoint.push_back(maskpointId[dynamask][point]);
                    willremovepointRight.push_back(mCurrentFrame.LeftIdtoRightId[maskpointId[dynamask][point]]);//记录要去除前后帧匹配上的点，但是实际是需要去去除当前帧msak的所有点
                    //　前后站匹配显示去除
                }
            } 
        }
        for (size_t i(0); i < mCurrentFrame.refermvKeysdyna.size(); ++i) {
            // 不在mask上
            if((int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[i].pt.y),
                    round(mCurrentFrame.refermvKeysdyna[i].pt.x)) == 255 || (int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[i].pt.y),
                            round(mCurrentFrame.refermvKeysdyna[i].pt.x)) == 0) {
                // 道路
                if((int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[i].pt.y),
                        round(mCurrentFrame.refermvKeysdyna[i].pt.x)) == 255) {

                    cv::Point2f pt1, pt2;
                    pt1.x = mCurrentFrame.refermvKeysdyna[i].pt.x - 5;
                    pt1.y = mCurrentFrame.refermvKeysdyna[i].pt.y - 5;
                    pt2.x = mCurrentFrame.refermvKeysdyna[i].pt.x + 5;
                    pt2.y = mCurrentFrame.refermvKeysdyna[i].pt.y + 5;
                  
                } else {
                    cv::Point2f pt1, pt2;
                    pt1.x = mCurrentFrame.refermvKeysdyna[i].pt.x - 5;
                    pt1.y = mCurrentFrame.refermvKeysdyna[i].pt.y - 5;
                    pt2.x = mCurrentFrame.refermvKeysdyna[i].pt.x + 5;
                    pt2.y = mCurrentFrame.refermvKeysdyna[i].pt.y + 5;
                  
                }

            } else if(dynamaskerea[(250 - (int)mCurrentFrame.maskleft.at<unsigned char>(round(mCurrentFrame.refermvKeysdyna[i].pt.y), round(mCurrentFrame.refermvKeysdyna[i].pt.x))) / 5]) {
                // 赋值为-1就是取消点信息，但不删除点
                mCurrentFrame.mvuLeft[i] = -1;
                mCurrentFrame.mvDepth[i] = -1;
                mCurrentFrame.mvuRight[i] = -1;
                mCurrentFrame.mvpMapPoints[i]  = static_cast<MapPoint *>(NULL);; //是否为地图点

                cv::Point2f pt1, pt2;
                pt1.x = mCurrentFrame.refermvKeysdyna[i].pt.x - 5;
                pt1.y = mCurrentFrame.refermvKeysdyna[i].pt.y - 5;
                pt2.x = mCurrentFrame.refermvKeysdyna[i].pt.x + 5;
                pt2.y = mCurrentFrame.refermvKeysdyna[i].pt.y + 5;
            } 
        }
        // txt save dynamic mask
        readWriteFile(dynamaskerea,mCurrentFrame.mnId);

    }

    /**
     * 保存动态区域的ｍａｓｋ
     * @param dynamaskerea
     * @param maskimg
     * @param savemask
     */
    Mat Tracking::SaveDynaMask(vector<bool> &dynamaskerea, Mat &maskimg) {
        Mat save_mask=maskimg.clone();
        for (int i = 0; i < save_mask.cols; i++) {
            for (int j = 0; j < save_mask.rows; j++) {
                if ((int) maskimg.at<unsigned char>(j, i) == 255 ){
                    save_mask.at<unsigned char>(j, i)=0;
                }
                else if (!dynamaskerea[(250 - (int) save_mask.at<unsigned char>(j, i)) / 5]){
                    save_mask.at<unsigned char>(j, i)=0;
                }
            }
        }
        return save_mask;
    }

    void Tracking::readWriteFile(vector<bool> &dynamaskerea,int index)
    {
        ofstream fout;
        fout.open("/home/MF/Desktop/dynamask.txt",ios::app);
        if(!fout.is_open())
        {
            return ;
        }
        char num[150];
        sprintf(num,"%06d",index);
        fout << num <<" ";
        for(int i=0;i<dynamaskerea.size();i++){
            if(dynamaskerea[i]){
                fout<<250-i*5<<" ";
            }
        }
        fout << endl;
        fout.close();
        return ;
    }

    // 计算三维向量v的反对称矩阵
    cv::Mat Tracking::SkewSymmetricMatrix(const cv::Mat &v) {
        return (cv::Mat_<float>(3, 3) <<
                0,              -v.at<float>(2),     v.at<float>(1),
                v.at<float>(2),               0,    -v.at<float>(0),
                -v.at<float>(1),  v.at<float>(0),                 0);
    }



    // Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
    // x' = P'X  x = PX
    // 它们都属于 x = aPX模型
    //                         |X|
    // |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
    // |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
    // |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
    // 采用DLT的方法：x叉乘PX = 0
    // |yp2 -  p1|     |0|
    // |p0 -  xp2| X = |0|
    // |xp1 - yp0|     |0|
    // 两个点:
    // |yp2   -  p1  |     |0|
    // |p0    -  xp2 | X = |0| ===> AX = 0
    // |y'p2' -  p1' |     |0|
    // |p0'   - x'p2'|     |0|
    // 变成程序中的形式：
    // |xp2  - p0 |     |0|
    // |yp2  - p1 | X = |0| ===> AX = 0
    // |x'p2'- p0'|     |0|
    // |y'p2'- p1'|     |0|
    // 然后就组成了一个四元一次正定方程组，求解呗

    //给定投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复3D坐标 (三角化)
    void Tracking::Triangulatelast2cur(
        const cv::KeyPoint &kp1,    //特征点, in reference frame
        const cv::KeyPoint &kp2,    //特征点, in current frame
        const cv::Mat &P1,          //投影矩阵P1
        const cv::Mat &P2,          //投影矩阵P2
        cv::Mat &x3D) {             //三维点

        //这个就是上面注释中的矩阵A
        cv::Mat A(4, 4, CV_32F);

        //构造参数矩阵A
        A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
        A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
        A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
        A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

        //奇异值分解的结果
        cv::Mat u, w, vt;
        //对系数矩阵A进行奇异值分解
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        //根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
        //别忘了我们更习惯用列向量来表示一个点的空间坐标
        x3D = vt.row(3).t();
        //为了符合其次坐标的形式，使最后一维为1
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(2);//z=1
    }


    // 给mask加颜色
    void Tracking::AddColor(Mat &imMask, Mat &outmat) {
        for (int row = 0; row < imMask.rows; row++) {
            for (int col = 0; col < imMask.cols; col++) {
                cv::Vec3b pixelVal = imMask.at<cv::Vec3b>(row, col);
                if(pixelVal[0] != 0) {
                    outmat.at<cv::Vec3b>(row, col) = MaskColor[((255 - pixelVal[0]) / 5)];
                }
            }
        }
    }


    // 计算当前帧组合中运动对象的速度
    /**
         * @brief CurrentSpeed 计算速度
         * @param cur            当前帧
         * @param last           上一帧
         * @param maskNumsmatch 匹配关系
         */
    void Tracking::CurrentSpeed(Frame &cur, Frame &last, vector<vector<int> > maskNumsmatch) {
        int rowsize = maskNumsmatch.size();
        Mat tempLeftimg = cur.imgLeft.clone();
        if(tempLeftimg.channels() == 1) {
            cvtColor(tempLeftimg, tempLeftimg, COLOR_GRAY2BGR);
        }
        vector<float> speed(rowsize, 0); //初始化速度
        for(int i = 0; i < rowsize; i++) {
            if(maskNumsmatch[i].front() == -1) {
                continue;
            }
        }
        waitKey(1);
    }


    // 使用距离判断是否是动态
    /**
     * @brief Tracking::WhetherDynawithDistance
     * @param maskMatch
     * @param num
     * @param curDis
     * @param frontDis
     * @param move
     * @return
     */

    bool Tracking::WhetherDynawithDistance(vector<vector<int> > &maskMatch, int &num, vector<vector<float> > &curDis, vector<vector<float> > &frontDis, float &move) {
        if(maskMatch[num].front() == -1) {
            return false;
        } else {
            if(abs(curDis[num].front() - frontDis[maskMatch[num].front()].front() - move) > curDis[num].front() * 0.03) {
                return true;
            } else {
                return false;
            }

        }
    }

    // 使用距离判断是否是动态,仅使用正前方的双目测距对象
    bool Tracking::WhetherDynawithDistance(vector<vector<int> > &maskMatch, int &num, vector<vector<float> > &curDis, vector<vector<float> > &frontDis, vector<vector<int> > &maskbord, float &move) {
        if(maskMatch[num].front() == -1) {
            return false;
        } else {
            if(abs(curDis[num].front() - frontDis[maskMatch[num].front()].front() - move) > move * 0.5) {
                float centerset = (maskbord[num][0] + maskbord[num][1]) / 2;
                if(centerset > width / 2 - 0.1 * width && centerset < width / 2 + 0.1 * width) {//此处可借鉴正前方分割原理车道，谋篇论文
                    return true;
                } else {
                    return false;
                }

            } else {
                return false;
            }

        }
    }


    // 处理vector<vector<int> > curMasknum2frontMasknums中筛选一一对应问题
    void Tracking::ChoicecurMaskOnebyOnefrontMask(vector<vector<int> > &curMasknum2frontMasknums, vector<vector<int> > &maskNumsmatch) {
        int rowsize = curMasknum2frontMasknums.size();
        maskNumsmatch.resize(rowsize, vector<int>());
        for(int i = 0; i < rowsize; i++) {
            if(curMasknum2frontMasknums[i].size() == 0) {
                maskNumsmatch[i].push_back(-1);
            } else {
                std::map<int, int> keyValue;
                std::vector<int>::iterator iter ;
                for(iter = curMasknum2frontMasknums[i].begin(); iter != curMasknum2frontMasknums[i].end(); ++iter) {
                    keyValue[*iter]++;
                }
                // 遍历找到列表value值最大的key
                std::map<int, int>::iterator keyiter;
                int maxValue = 0;//数量
                int key = -1;//mask值
                for(keyiter = keyValue.begin(); keyiter != keyValue.end(); ++keyiter) {
                    int temp = -1;
                    temp = keyiter->second; //数量
                    if(temp > maxValue) {
                        key = keyiter->first;
                    }
                }
                maskNumsmatch[i].push_back(key);




            }
        }
    }


    // 计算mask列边界，当然加上行边界也可以的嘛
    void Tracking::Calculmaskborder(cv::Mat &maskborder, int &maskclassnums, vector<vector<int> > &border) {
        if(maskclassnums == 0) {
            return;
        } else {
            //循环计算各个掩码对应的最小最大值边界
            // step1 创建对应mask 类的最小最大值
            //初始化一个最大最小值
            for(int i = 0; i < maskclassnums; i++) {
//                border[i].reserve(2);
                border[i].push_back(maskborder.cols); //最左边界
                border[i].push_back(0);//最后边界
            }
            // 计算最大最小值
            for(int i = 0, rows = maskborder.rows; i < rows; i++) {
                for(int j = 0, cols = maskborder.cols; j < cols; j++) {
                    if((int)maskborder.at<unsigned char>(i, j) == 255 || (int)maskborder.at<unsigned char>(i, j) == 0) {
                        // 如果不在mask内，后续在这里加记录mask外的点
                        continue;
                    } else {
                        // 在mask上输出点的mask点的值
                        int maskvalue = (250 - (int)maskborder.at<unsigned char>(i, j)) / 5;
                        if(maskvalue >= 0 && maskvalue < maskclassnums) {
                            if(border[maskvalue][0] > j) {
                                border[maskvalue][0] = j;
                            }
                            if(border[maskvalue][1] < j) {
                                border[maskvalue][1] = j;
                            }
                        }
                    }
                }
            }
        }
        

    }

    // 计算低于百分比阈值的平均数
    float Tracking::CalcullowerThaverdis(vector<float> &pointsdistance, float &th) {
        int psize = pointsdistance.size();
        if(psize == 0) {
            return -1;
        }
        sort(pointsdistance.begin(), pointsdistance.end());
        // 区间拷贝,利用构造函数
        std:: vector<float> lowerTh(pointsdistance.begin(), pointsdistance.begin() + round(pointsdistance.size()*th));
        return Calcuaverdistance(lowerTh);
    }

    // 选去极线距离某个阈值对应的距离
    float Tracking::ChoiceoneEpipolardistance(vector<float> &pointsdistance, float &th) {
        int psize = pointsdistance.size();
        if(psize == 0) {
            return -1;
        }
        sort(pointsdistance.begin(), pointsdistance.end());
        return pointsdistance[round(psize * th)];
    }

//　计算平均距离
    float Tracking::Calcuaverdistance(vector<float> &savedistance) {
        if(savedistance.empty()) {
            return INT_MIN;
        } else {
            return std::accumulate(std::begin(savedistance), std::end(savedistance), 0.0) / savedistance.size();
        }
    }

    // 计算ｆ*vec(x1`*x2)
    float Tracking::CalculFPP(cv::Mat &F, cv::Point2f &p0, cv::Point2f &p1) {
        return F.at<float>(0, 0) * p1.x * p0.x + F.at<float>(0, 1) * p1.x * p0.y + F.at<float>(0, 2) * p1.x +
               F.at<float>(1, 0) * p1.y * p0.x + F.at<float>(1, 1) * p1.y * p0.y + F.at<float>(1, 2) * p1.y +
               F.at<float>(2, 0) * p0.x + F.at<float>(2, 1) * p0.y + F.at<float>(2, 2);
    }

    float Tracking::Calcultwopointdistance(Point2f &pfront, Point2f &pafter) {
        return sqrt(pow((pfront.x - pafter.x), 2) + pow((pfront.y - pafter.y), 2));

    }

    /**
     * @brief Tracking::CalculEpipolardistance 计算极距离
     * @param p2    上一帧的点
     * @param p1    当前帧的帧
     * @param F     基础矩阵
     * @return
     */
    float Tracking::CalculEpipolardistance(cv::Point2f &p2, cv::Point2f &p1, cv::Mat &F) {
        if(!F.data) {
            return -1;
        } else {
            //  step1 归一化点为mat
            Mat matp2, matp1;
            matp2 = (cv::Mat_<float>(1, 3) << p2.x, p2.y, 1);
            matp1 = (cv::Mat_<float>(3, 1) << p1.x, p1.y, 1);
            // step2 计算极线l=F*p1
            Mat l = F * matp1;
            // step3 对极限归一化
            Mat norl = (cv::Mat_<float>(3, 1) << l.at<float>(0, 0) / l.at<float>(2, 0), l.at<float>(1, 0) / l.at<float>(2, 0), 1);
            // step4 计算分子　｜matp2*F*matp1｜
            Mat tempd0 = matp2 * F * matp1;
            float d0 = abs(tempd0.at<float>(0, 0));
            // step5 计算分母
            float d = sqrt(pow(norl.at<float>(0, 0), 2) + pow(norl.at<float>(1, 0), 2));
            return d0 / d;
        }
    }

    void Tracking::FrontandbackKeyframematching() {
        // 跟踪初始化时前两帧之间的匹配
        // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
        std::vector<cv::Point2f> MFlastKeypoints0, MFcurrentKeypoints;
        // MFlast0  记录"上一帧"所有特征点
        MFlastKeypoints0.resize(MFLast1KeyFrame.mvKeysUn.size());
        //MFcurrent 记录"上一帧"所有特征点
        MFcurrentKeypoints.resize(MFLast0KeyFrame.mvKeysUn.size());
        //　取出前一帧点
        for(size_t i = 0; i < MFLast1KeyFrame.mvKeysUn.size(); i++) {
            MFlastKeypoints0[i] = MFLast1KeyFrame.mvKeysUn[i].pt;
        }
        //　取出当前帧的点
        for(size_t i = 0; i < MFLast0KeyFrame.mvKeysUn.size(); i++) {
            MFcurrentKeypoints[i] = MFLast0KeyFrame.mvKeysUn[i].pt;
        }
      //在mInitialFrame与mCurrentFrame中找匹配的特征点对
        ORBmatcher MFmatcher(
            0.9,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
            true);      //检查特征点的方向
       MFmatcher.MFSearchForSegmentregion(MFLast1KeyFrame, MFLast0KeyFrame, MFlastKeypoints0, FrontbackKeyMatches, FrontbackoutKeyMatches, 100);
        // 测试匹配的点对
        int tempnumforfb = 0;
        for(int i = 0, fbnum = FrontbackKeyMatches.size(); i < fbnum; i++) {
            if(FrontbackKeyMatches[i] == -1) {
                continue;
            } else {
                tempnumforfb++;
            }
        }
        cout << endl << " 前后关键帧一共匹配点数　" << tempnumforfb << endl;
      
    }




    void Tracking::MonocularInitialization() {
        if(!mpInitializer) {
            // Set Reference Frame
            if(mCurrentFrame.mvKeys.size() > 100) {
                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for(size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++) {
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
                }
                if(mpInitializer) {
                    delete mpInitializer;
                }
                mpInitializer =  new Initializer(mCurrentFrame, 1.0, 200);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                if (mSensor == System::IMU_MONOCULAR) {
                    if(mpImuPreintegratedFromLastKF) {
                        delete mpImuPreintegratedFromLastKF;
                    }
                    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
                }
                return;
            }
        } else {
            if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0))) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }
            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
            // Check if there are enough correspondences
            if(nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }
            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
            if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
                for(size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if(mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }
                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);
                CreateInitialMapMonocular();
                // Just for video
                // bStepByStep = true;
            }
        }
    }



    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        if(mSensor == System::IMU_MONOCULAR) {
            pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
        }
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();
        // Insert KFs in the map
        mpAtlas->AddKeyFrame(pKFini);
        mpAtlas->AddKeyFrame(pKFcur);
        for(size_t i = 0; i < mvIniMatches.size(); i++) {
            if(mvIniMatches[i] < 0) {
                continue;
            }
            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
            //Add to Map
            mpAtlas->AddMapPoint(pMP);
        }
        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();
        std::set<MapPoint *> sMPs;
        sMPs = pKFini->GetMapPoints();
        // Bundle Adjustment
        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
        Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);
        pKFcur->PrintPointDistribution();
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth;
        if(mSensor == System::IMU_MONOCULAR) {
            invMedianDepth = 4.0f / medianDepth;    // 4.0f
        } else {
            invMedianDepth = 1.0f / medianDepth;
        }
        if(medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) { // TODO Check, originally 100 tracks
            Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
            mpSystem->ResetActiveMap();
            return;
        }
        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);
        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for(size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if(vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
                pMP->UpdateNormalAndDepth();
            }
        }
        if (mSensor == System::IMU_MONOCULAR) {
            pKFcur->mPrevKF = pKFini;
            pKFini->mNextKF = pKFcur;
            pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
        }
        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;
        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        mnLastRelocFrameId = mInitialFrame.mnId;
        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;
        // Compute here initial velocity
        vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames();
        cv::Mat deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
        mVelocity = cv::Mat();
        Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0, 3).colRange(0, 3)));
        double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
        phi *= aux;
        mLastFrame = Frame(mCurrentFrame);
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);
        mState = OK;
        initID = pKFcur->mnId;
    }


    void Tracking::CreateMapInAtlas() {
        mnLastInitFrameId = mCurrentFrame.mnId;
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR) {
            mpAtlas->SetInertialSensor();
        }
        mbSetInit = false;
        mnInitialFrameId = mCurrentFrame.mnId + 1;
        mState = NO_IMAGES_YET;
        // Restart the variable with information about the last KF
        mVelocity = cv::Mat();
        mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
        Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
        mbVO = false; // Init value for know if there are enough MapPoints in the last KF
        if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) {
            if(mpInitializer) {
                delete mpInitializer;
            }
            mpInitializer = static_cast<Initializer *>(NULL);
        }
        if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO ) && mpImuPreintegratedFromLastKF) {
            delete mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
        }
        if(mpLastKeyFrame) {
            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        }
        if(mpReferenceKF) {
            mpReferenceKF = static_cast<KeyFrame *>(NULL);
        }
        mLastFrame = Frame();
        mCurrentFrame = Frame();
        mvIniMatches.clear();
        mbCreatedMap = true;
    }

    void Tracking::CheckReplacedInLastFrame() {
        for(int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if(pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if(pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }


    bool Tracking::TrackReferenceKeyFrame() {
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        if(nmatches < 15) {
            cout << "TRACK_REF_KF: Less than 15 matches!!\n";
            return false;
        }
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw);
        //mCurrentFrame.PrintPointDistribution();
        Optimizer::PoseOptimization(&mCurrentFrame);
        // Discard outliers
        int nmatchesMap = 0;
        for(int i = 0; i < mCurrentFrame.N; i++) {
            //if(i >= mCurrentFrame.Nleft) break;
            if(mCurrentFrame.mvpMapPoints[i]) {
                if(mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if(i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    } else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
                    nmatchesMap++;
                }
            }
        }
        // TODO check these conditions
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
            return true;
        } else {
            return nmatchesMap >= 10;
        }
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * pRef->GetPose());
        if(mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking) {
            return;
        }
        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for(int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if(z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }
        if(vDepthIdx.empty()) {
            return;
        }
        sort(vDepthIdx.begin(), vDepthIdx.end());
        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for(size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if(!pMP) {
                bCreateNew = true;
            } else if(pMP->Observations() < 1) {
                bCreateNew = true;
            }
            if(bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
                mLastFrame.mvpMapPoints[i] = pNewMP;
                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }
            if(vDepthIdx[j].first > mThDepth && nPoints > 100) {
                break;
            }
        }
    }

    bool Tracking::TrackWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();
        if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
            // Predict ste with IMU if it is initialized and it doesnt need reset
            PredictStateIMU();
            return true;
        } else {
            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
        }
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        // Project points seen in previous frame
        int th;
        if(mSensor == System::STEREO) {
            th = 7;
        } else {
            th = 15;
        }
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
        // If few matches, uses a wider window search
        if(nmatches < 20) {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
        }
        if(nmatches < 20) {
            Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
                return true;
            } else {
                return false;
            }
        }
        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);
        // Discard outliers
        int nmatchesMap = 0;
        for(int i = 0; i < mCurrentFrame.N; i++) {
            if(mCurrentFrame.mvpMapPoints[i]) {
                if(mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if(i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    } else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
                    nmatchesMap++;
                }
            }
        }
        if(mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
            return true;
        } else {
            return nmatchesMap >= 10;
        }
    }

    bool Tracking::TrackLocalMap() {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        mTrackedFr++;
        UpdateLocalMap();
        SearchLocalPoints();
        // TOO check outliers before PO
        int aux1 = 0, aux2 = 0;
        for(int i = 0; i < mCurrentFrame.N; i++)
            if( mCurrentFrame.mvpMapPoints[i]) {
                aux1++;
                if(mCurrentFrame.mvbOutlier[i]) {
                    aux2++;
                }
            }
        int inliers;
        if (!mpAtlas->isImuInitialized()) {
            Optimizer::PoseOptimization(&mCurrentFrame);
        } else {
            if(mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU) {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            } else {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                if(!mbMapUpdated) { //  && (mnMatchesInliers>30))
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                } else {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }
        aux1 = 0, aux2 = 0;
        for(int i = 0; i < mCurrentFrame.N; i++)
            if( mCurrentFrame.mvpMapPoints[i]) {
                aux1++;
                if(mCurrentFrame.mvbOutlier[i]) {
                    aux2++;
                }
            }
        mnMatchesInliers = 0;
        // Update MapPoints Statistics
        for(int i = 0; i < mCurrentFrame.N; i++) {
            if(mCurrentFrame.mvpMapPoints[i]) {
                if(!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if(!mbOnlyTracking) {
                        if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
                            mnMatchesInliers++;
                        }
                    } else {
                        mnMatchesInliers++;
                    }
                } else if(mSensor == System::STEREO) {
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }
        }
        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
        if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50) {
            return false;
        }
        if((mnMatchesInliers > 10) && (mState == RECENTLY_LOST)) {
            return true;
        }
        if (mSensor == System::IMU_MONOCULAR) {
            if(mnMatchesInliers < 15) {
                return false;
            } else {
                return true;
            }
        } else if (mSensor == System::IMU_STEREO) {
            if(mnMatchesInliers < 15) {
                return false;
            } else {
                return true;
            }
        } else {
            if(mnMatchesInliers < 30) {
                return false;
            } else {
                return true;
            }
        }
    }

    bool Tracking::NeedNewKeyFrame() {
        if(((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && !mpAtlas->GetCurrentMap()->isImuInitialized()) {
            if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25) {
                return true;
            } else if (mVelocity.at<float>(2, 0) > 0.010 || mVelocity.at<float>(2, 0) < -0.010 ) {
                if(mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.10) {
                    return true;
                } else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25) {
                    return true;
                }
            } else {
                return false;
            }
        }
        if(mbOnlyTracking) {
            return false;
        }
        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
            return false;
        }
        // Return false if IMU is initialazing
        if (mpLocalMapper->IsInitializing()) {
            return false;
        }
        const int nKFs = mpAtlas->KeyFramesInMap();
        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
            return false;
        }
        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if(nKFs <= 2) {
            nMinObs = 2;
        }
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0; //现有地图中,可以被当前帧观测到的地图点数目
        int nTotal = 0;     //当前帧中可以添加到地图中的地图点数量 , 总的可以添加mappoints数
        if(mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
            int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
            for(int i = 0; i < N; i++) {
                if(mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                        nTrackedClose++;
                    } else {
                        nNonTrackedClose++;
                    }
                }
            }
        }
        bool bNeedToInsertClose;
        bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);
        // like slam2
        // ratioMap: 计算这个比例,当前帧中观测到的地图点数目和当前帧中总共的地图点数目之比.这个值越接近1越好,越接近0说明跟踪上的地图点太少,tracking is weak
     // Thresholds
        float thRefRatio ;
        //if(mVelocity.at<float>(2,0)> 0.010 || mVelocity.at<float>(2,0) < -0.010 )
        if(fabs(mVelocity.at<float>(2, 0)) > 0.01) {
            thRefRatio = 0.95f;
        } else {
            thRefRatio = 0.75f;
        }
        if(nKFs < 2) {
            thRefRatio = 0.4f;
        }
        if(mSensor == System::MONOCULAR) {
            thRefRatio = 0.9f;
        }
        if(mpCamera2) {
            thRefRatio = 0.75f;
        }
        if(mSensor == System::IMU_MONOCULAR) {
            if(mnMatchesInliers > 350) { // Points tracked from the local map
                thRefRatio = 0.75f;
            } else {
                thRefRatio = 0.90f;
            }
        }
        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO &&
                         (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose ) ;
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose )) && mnMatchesInliers > 15 );
        // Temporal condition for Inertial cases
        bool c3 = false;
        if(mpLastKeyFrame) {
            if (mSensor == System::IMU_MONOCULAR) {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5) {
                    c3 = true;
                }
            } else if (mSensor == System::IMU_STEREO) {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5) {
                    c3 = true;
                }
            }
        }
        bool c4 = false;
        if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR))) { // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
            c4 = true;
        } else {
            c4 = false;
        }
        if(((c1a || c1b || c1c) && c2) || c3 || c4) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if(bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if(mSensor != System::MONOCULAR  && mSensor != System::IMU_MONOCULAR) {
                    if(mpLocalMapper->KeyframesInQueue() < 3) {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    void Tracking::CreateNewKeyFrame() {
        if(mpLocalMapper->IsInitializing()) {
            return;
        }
        if(!mpLocalMapper->SetNotStop(true)) {
            return;
        }
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        if(mpAtlas->isImuInitialized()) {
            pKF->bImu = true;
        }
        pKF->SetNewBias(mCurrentFrame.mImuBias);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;
        if(mpLastKeyFrame) {
            pKF->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKF;
        } else {
            Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);
        }
        // Reset preintegration from last KF (Create new object)
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
        }
        if(mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) { // TODO check if incluide imu_stereo
            mCurrentFrame.UpdatePoseMatrices(); 
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            int maxPoint = 100;
            if(mSensor == System::IMU_STEREO) {
                maxPoint = 100;
            }
            vector<pair<float, int> > vDepthIdx;
            int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
            vDepthIdx.reserve(mCurrentFrame.N);
            for(int i = 0; i < N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if(z > 0) {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }
            if(!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());
                int nPoints = 0;
                for(size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;
                    bool bCreateNew = false;
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if(!pMP) {
                        bCreateNew = true;
                    } else if(pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                    if(bCreateNew) {
                        cv::Mat x3D;
                        if(mCurrentFrame.Nleft == -1) {
                            x3D = mCurrentFrame.UnprojectStereo(i);
                        } else {
                            x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                        }
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKF, i);
                        //Check if it is a stereo observation in order to not
                        //duplicate mappoints
                        if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
                            mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                            pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                            pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        }
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);
                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++; // TODO check ???
                    }
                    if(vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
                        break;
                    }
                }
                Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
            }
        }
        mpLocalMapper->InsertKeyFrame(pKF);
        mpLocalMapper->SetNotStop(false);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints() {
        // Do not search map points already matched
        for(vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if(pMP) {
                if(pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                    pMP->mbTrackInViewR = false;
                }
            }
        }
        int nToMatch = 0;
        // Project points in frame and check its visibility
        for(vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) {
                continue;
            }
            if(pMP->isBad()) {
                continue;
            }
            // Project (this fills MapPoint variables for matching)
            if(mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++;
            }
            if(pMP->mbTrackInView) {
                mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
            }
        }
        if(nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if(mSensor == System::RGBD) {
                th = 3;
            }
            if(mpAtlas->isImuInitialized()) {
                if(mpAtlas->GetCurrentMap()->GetIniertialBA2()) {
                    th = 2;
                } else {
                    th = 3;
                }
            } else if(!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
                th = 10;
            }
            // If the camera has been relocalised recently, perform a coarser search
            if(mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                th = 5;
            }
            if(mState == LOST || mState == RECENTLY_LOST) { // Lost for less than 1 second
                th = 15;    // 15
            }
            int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
        }
    }

    void Tracking::UpdateLocalMap() {
        // This is for visualization
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();
        int count_pts = 0;
        for(vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            for(vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if(!pMP) {
                    continue;
                }
                if(pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) {
                    continue;
                }
                if(!pMP->isBad()) {
                    count_pts++;
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2)) {
            for(int i = 0; i < mCurrentFrame.N; i++) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP) {
                    if(!pMP->isBad()) {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for(map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++) {
                            keyframeCounter[it->first]++;
                        }
                    } else {
                        mCurrentFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        } else {
            for(int i = 0; i < mLastFrame.N; i++) {
                // Using lastframe since current frame has not matches yet
                if(mLastFrame.mvpMapPoints[i]) {
                    MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                    if(!pMP) {
                        continue;
                    }
                    if(!pMP->isBad()) {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for(map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++) {
                            keyframeCounter[it->first]++;
                        }
                    } else {
                        // MODIFICATION
                        mLastFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }
        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);
        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());
        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for(map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++) {
            KeyFrame *pKF = it->first;
            if(pKF->isBad()) {
                continue;
            }
            if(it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }
            mvpLocalKeyFrames.push_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for(vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if(mvpLocalKeyFrames.size() > 80) { // 80
                break;
            }
            KeyFrame *pKF = *itKF;
            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
            for(vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if(!pNeighKF->isBad()) {
                    if(pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }
            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for(set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if(!pChildKF->isBad()) {
                    if(pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }
            KeyFrame *pParent = pKF->GetParent();
            if(pParent) {
                if(pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }
        // Add 10 last temporal KFs (mainly for IMU)
        if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mvpLocalKeyFrames.size() < 80) {
            KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;
            const int Nd = 20;
            for(int i = 0; i < Nd; i++) {
                if (!tempKeyFrame) {
                    break;
                }
                if(tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(tempKeyFrame);
                    tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    tempKeyFrame = tempKeyFrame->mPrevKF;
                }
            }
        }
        if(pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization() {
        Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();
        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());
        if(vpCandidateKFs.empty()) {
            Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
            return false;
        }
        const int nKFs = vpCandidateKFs.size();
        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);
        vector<MLPnPsolver *> vpMLPnPsolvers;
        vpMLPnPsolvers.resize(nKFs);
        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);
        int nCandidates = 0;
        for(int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if(pKF->isBad()) {
                vbDiscarded[i] = true;
            } else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if(nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991); //This solver needs at least 6 points
                    vpMLPnPsolvers[i] = pSolver;
                }
            }
        }
        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);
        while(nCandidates > 0 && !bMatch) {
            for(int i = 0; i < nKFs; i++) {
                if(vbDiscarded[i]) {
                    continue;
                }
                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;
                MLPnPsolver *pSolver = vpMLPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
                // If Ransac reachs max. iterations discard keyframe
                if(bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }
                // If a Camera Pose is computed, optimize
                if(!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);
                    set<MapPoint *> sFound;
                    const int np = vbInliers.size();
                    for(int j = 0; j < np; j++) {
                        if(vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else {
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                        }
                    }
                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                    if(nGood < 10) {
                        continue;
                    }
                    for(int io = 0; io < mCurrentFrame.N; io++)
                        if(mCurrentFrame.mvbOutlier[io]) {
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
                        }
                    // If few inliers, search by projection in a coarse window and optimize again
                    if(nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                        if(nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if(nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for(int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if(mCurrentFrame.mvpMapPoints[ip]) {
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                    }
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                                // Final optimization
                                if(nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                    for(int io = 0; io < mCurrentFrame.N; io++)
                                        if(mCurrentFrame.mvbOutlier[io]) {
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                        }
                                }
                            }
                        }
                    }
                    // If the pose is supported by enough inliers stop ransacs and continue
                    if(nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }
        }
        if(!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            cout << "Relocalized!!" << endl;
            return true;
        }
    }

    void Tracking::Reset(bool bLocMap) {
        Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);
        if(mpViewer) {
            mpViewer->RequestStop();
            while(!mpViewer->isStopped()) {
                usleep(3000);
            }
        }
        // Reset Local Mapping
        if (!bLocMap) {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestReset();
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }
        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        // Clear BoW Database
        Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clear();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearAtlas();
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR) {
            mpAtlas->SetInertialSensor();
        }
        mnInitialFrameId = 0;
        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;
        if(mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }
        mbSetInit = false;
        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();
        mCurrentFrame = Frame();
        mnLastRelocFrameId = 0;
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();
        if(mpViewer) {
            mpViewer->Release();
        }
        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    void Tracking::ResetActiveMap(bool bLocMap) {
        Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
        if(mpViewer) {
            mpViewer->RequestStop();
            while(!mpViewer->isStopped()) {
                usleep(3000);
            }
        }
        Map *pMap = mpAtlas->GetCurrentMap();
        if (!bLocMap) {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestResetActiveMap(pMap);
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }
        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        // Clear BoW Database
        Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearMap();
        //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
        //Frame::nNextId = mnLastInitFrameId;
        mnLastInitFrameId = Frame::nNextId;
        mnLastRelocFrameId = mnLastInitFrameId;
        mState = NO_IMAGES_YET; //NOT_INITIALIZED;
        if(mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }
        list<bool> lbLost;
        unsigned int index = mnFirstFrameId;
        cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
        for(Map *pMap : mpAtlas->GetAllMaps()) {
            if(pMap->GetAllKeyFrames().size() > 0) {
                if(index > pMap->GetLowerKFID()) {
                    index = pMap->GetLowerKFID();
                }
            }
        }
        int num_lost = 0;
        cout << "mnInitialFrameId = " << mnInitialFrameId << endl;
        for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++) {
            if(index < mnInitialFrameId) {
                lbLost.push_back(*ilbL);
            } else {
                lbLost.push_back(true);
                num_lost += 1;
            }
            index++;
        }
        cout << num_lost << " Frames had been set to lost" << endl;
        mlbLost = lbLost;
        mnInitialFrameId = mCurrentFrame.mnId;
        mnLastRelocFrameId = mCurrentFrame.mnId;
        mCurrentFrame = Frame();
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();
        if(mpViewer) {
            mpViewer->Release();
        }
        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    vector<MapPoint *> Tracking::GetLocalMapMPS() {
        return mvpLocalMapPoints;
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);
        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);
        mbf = fSettings["Camera.bf"];
        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

    void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame) {
        Map *pMap = pCurrentKeyFrame->GetMap();
        unsigned int index = mnFirstFrameId;
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for(list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++) {
            if(*lbL) {
                continue;
            }
            KeyFrame *pKF = *lRit;
            while(pKF->isBad()) {
                pKF = pKF->GetParent();
            }
            if(pKF->GetMap() == pMap) {
                (*lit).rowRange(0, 3).col(3) = (*lit).rowRange(0, 3).col(3) * s;
            }
        }
        mLastBias = b;
        mpLastKeyFrame = pCurrentKeyFrame;
        mLastFrame.SetNewBias(mLastBias);
        mCurrentFrame.SetNewBias(mLastBias);
        cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
        cv::Mat twb1;
        cv::Mat Rwb1;
        cv::Mat Vwb1;
        float t12;
        while(!mCurrentFrame.imuIsPreintegrated()) {
            usleep(500);
        }
        if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId) {
            mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                          mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                          mLastFrame.mpLastKeyFrame->GetVelocity());
        } else {
            twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
            t12 = mLastFrame.mpImuPreintegrated->dT;
            mLastFrame.SetImuPoseVelocity(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                          twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                          Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }
        if (mCurrentFrame.mpImuPreintegrated) {
            twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
            t12 = mCurrentFrame.mpImuPreintegrated->dT;
            mCurrentFrame.SetImuPoseVelocity(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                             twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                             Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }
        mnFirstImuFrameId = mCurrentFrame.mnId;
    }

    //　计算关键帧Ｆ矩阵
    cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;
        cv::Mat t12x = Converter::tocvSkewMatrix(t12);
        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;
        return K1.t().inv() * t12x * R12 * K2.inv();
    }


    // 普通帧计算T矩阵
    cv::Mat Tracking::ComputeframeT12(Frame *pKF1, Frame *pKF2) {
        if(pKF1->mTcw.empty()) {
            return pKF1->mTcw;
        } else {
            cv::Mat R1w = pKF1->mTcw.rowRange(0, 3).colRange(0, 3).clone();
            cv::Mat t1w = pKF1->mTcw.rowRange(0, 3).col(3).clone();
            cv::Mat R2w = pKF2->mTcw.rowRange(0, 3).colRange(0, 3).clone();//这个是个空的
            cv::Mat t2w = pKF2->mTcw.rowRange(0, 3).col(3).clone();
            cv::Mat R12 = R1w * R2w.t();
            cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;
            cv::Mat T = cv::Mat::eye(4, 4, CV_32F);
            R12.copyTo(T.rowRange(0, 3).colRange(0, 3));
            t12.copyTo(T.rowRange(0, 3).col(3));
            return T;
        }
    }

    // 普通帧计算Ｆ矩阵
    cv::Mat Tracking::ComputeframeF12(Frame *pKF1, Frame *pKF2) {
        if(pKF1->mTcw.empty()) {
            return pKF1->Fcur;
        } else {
            cv::Mat R1w = pKF1->mTcw.rowRange(0, 3).colRange(0, 3).clone();
            cv::Mat t1w = pKF1->mTcw.rowRange(0, 3).col(3).clone();
            cv::Mat R2w = pKF2->mTcw.rowRange(0, 3).colRange(0, 3).clone();
            cv::Mat t2w = pKF2->mTcw.rowRange(0, 3).col(3).clone();
            cv::Mat R12 = R1w * R2w.t();
            cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;
            cv::Mat t12x = Converter::tocvSkewMatrix(t12);
            const cv::Mat &K1 = pKF1->mK;
            const cv::Mat &K2 = pKF2->mK;
            return K1.t().inv() * t12x * R12 * K2.inv();
        }
    }

    // 普通帧计算E矩阵
    cv::Mat Tracking::ComputeframeE12(Frame *pKF1, Frame *pKF2) {
        cv::Mat R1w = pKF1->mTcw.rowRange(0, 3).colRange(0, 3).clone();
        cv::Mat t1w = pKF1->mTcw.rowRange(0, 3).col(3).clone();
        cv::Mat R2w = pKF2->mTcw.rowRange(0, 3).colRange(0, 3).clone();
        cv::Mat t2w = pKF2->mTcw.rowRange(0, 3).col(3).clone();
        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;
        cv::Mat t12x = Converter::tocvSkewMatrix(t12);
        return t12x * R12 ;
    }


    void Tracking::CreateNewMapPoints() {
        // Retrieve neighbor keyframes in covisibility graph
        const vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
        ORBmatcher matcher(0.6, false);
        cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));
        cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();
        const float &fx1 = mpLastKeyFrame->fx;
        const float &fy1 = mpLastKeyFrame->fy;
        const float &cx1 = mpLastKeyFrame->cx;
        const float &cy1 = mpLastKeyFrame->cy;
        const float &invfx1 = mpLastKeyFrame->invfx;
        const float &invfy1 = mpLastKeyFrame->invfy;
        const float ratioFactor = 1.5f * mpLastKeyFrame->mfScaleFactor;
        int nnew = 0;
        // Search matches with epipolar restriction and triangulate
        for(size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF2 = vpKFs[i];
            if(pKF2 == mpLastKeyFrame) {
                continue;
            }
            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);
            if((mSensor != System::MONOCULAR) || (mSensor != System::IMU_MONOCULAR)) {
                if(baseline < pKF2->mb) {
                    continue;
                }
            } else {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline / medianDepthKF2;
                if(ratioBaselineDepth < 0.01) {
                    continue;
                }
            }
            // Compute Fundamental Matrix
            // Step 4：根据两个关键帧的位姿计算它们之间的基本矩阵
            cv::Mat F12 = ComputeF12(mpLastKeyFrame, pKF2);
            // Search matches that fullfil epipolar constraint
            // Step 5：通过极线约束限制匹配时的搜索范围，进行特征点匹配
            vector<pair<size_t, size_t> > vMatchedIndices;
            matcher.SearchForTriangulation(mpLastKeyFrame, pKF2, F12, vMatchedIndices, false);
            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));
            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;
            // Triangulate each match
            const int nmatches = vMatchedIndices.size();
            for(int ikp = 0; ikp < nmatches; ikp++) {
                const int &idx1 = vMatchedIndices[ikp].first;
                const int &idx2 = vMatchedIndices[ikp].second;
                const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
                const float kp1_ur = mpLastKeyFrame->mvuRight[idx1];
                bool bStereo1 = kp1_ur >= 0;
                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                const float kp2_ur = pKF2->mvuRight[idx2];
                bool bStereo2 = kp2_ur >= 0;
                // Check parallax between rays
                // step 6.2：利用匹配点反投影得到视差角
                // 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);
                // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;
                // 这个就是求向量之间角度公式
                const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
                float cosParallaxStereo = cosParallaxRays + 1;
                float cosParallaxStereo1 = cosParallaxStereo;
                float cosParallaxStereo2 = cosParallaxStereo;
                if(bStereo1) {
                    cosParallaxStereo1 = cos(2 * atan2(mpLastKeyFrame->mb / 2, mpLastKeyFrame->mvDepth[idx1]));
                } else if(bStereo2) {
                    cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
                }
                cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);
                cv::Mat x3D;
                if(cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
                    // Linear Triangulation Method
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);
                    cv::Mat w, u, vt;
                    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
                    x3D = vt.row(3).t();
                    if(x3D.at<float>(3) == 0) {
                        continue;
                    }
                    // Euclidean coordinates
                    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                } else if(bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
                    x3D = mpLastKeyFrame->UnprojectStereo(idx1);
                } else if(bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
                    x3D = pKF2->UnprojectStereo(idx2);
                } else {
                    continue;    //No stereo and very low parallax
                }
                cv::Mat x3Dt = x3D.t();
                //Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
                if(z1 <= 0) {
                    continue;
                }
                float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
                if(z2 <= 0) {
                    continue;
                }
                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
                const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
                const float invz1 = 1.0 / z1;
                if(!bStereo1) {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    if((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) {
                        continue;
                    }
                } else {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float u1_r = u1 - mpLastKeyFrame->mbf * invz1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    float errX1_r = u1_r - kp1_ur;
                    if((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1) {
                        continue;
                    }
                }
                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
                const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
                const float invz2 = 1.0 / z2;
                if(!bStereo2) {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    if((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) {
                        continue;
                    }
                } else {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float u2_r = u2 - mpLastKeyFrame->mbf * invz2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    float errX2_r = u2_r - kp2_ur;
                    if((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2) {
                        continue;
                    }
                }
                //Check scale consistency
                cv::Mat normal1 = x3D - Ow1;
                float dist1 = cv::norm(normal1);
                cv::Mat normal2 = x3D - Ow2;
                float dist2 = cv::norm(normal2);
                if(dist1 == 0 || dist2 == 0) {
                    continue;
                }
                const float ratioDist = dist2 / dist1;
                const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];
                if(ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) {
                    continue;
                }
                // Triangulation is succesfull
                MapPoint *pMP = new MapPoint(x3D, mpLastKeyFrame, mpAtlas->GetCurrentMap());
                pMP->AddObservation(mpLastKeyFrame, idx1);
                pMP->AddObservation(pKF2, idx2);
                mpLastKeyFrame->AddMapPoint(pMP, idx1);
                pKF2->AddMapPoint(pMP, idx2);
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
                mpAtlas->AddMapPoint(pMP);
                nnew++;
            }
        }
        TrackReferenceKeyFrame();
    }

    void Tracking::NewDataset() {
        mnNumDataset++;
    }

    int Tracking::GetNumberDataset() {
        return mnNumDataset;
    }

    int Tracking::GetMatchesInliers() {
        return mnMatchesInliers;
    }

} //namespace ORB_SLAM
