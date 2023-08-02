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


#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"
#include <algorithm>
#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <cmath>
#include "Drawkp.h"
#include <thread>
#include "Points.h"
#include <time.h>
#include<fstream>
#include<iomanip>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
    cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

    Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false) {
    }


//Copy Constructor
    Frame::Frame(const Frame &frame)
        : mpcpi(frame.mpcpi), mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
          mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
          mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
          mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
          mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
          mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
          mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
          mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
          mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
          mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
          mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
          mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
          mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
          monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
          mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
          mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()),
          mTimeStereoMatch(frame.mTimeStereoMatch), mTimeORB_Ext(frame.mTimeORB_Ext) {
        for(int i = 0; i < FRAME_GRID_COLS; i++)
            for(int j = 0; j < FRAME_GRID_ROWS; j++) {
                mGrid[i][j] = frame.mGrid[i][j];
                if(frame.Nleft > 0) {
                    mGridRight[i][j] = frame.mGridRight[i][j];
                }
            }
        if(!frame.mTcw.empty()) {
            SetPose(frame.mTcw);
        }
        if(!frame.mVw.empty()) {
            mVw = frame.mVw.clone();
        }
        mmProjectPoints = frame.mmProjectPoints;
        mmMatchedInImage = frame.mmMatchedInImage;
    }

//　stereo .
    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF, const IMU::Calib &ImuCalib)
        : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false),
          mpCamera(pCamera), mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0) {
        // Frame ID
        mnId = nNextId++;
        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        // ORB extraction
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);
        threadLeft.join();
        threadRight.join();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
        mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
        //mvKeys, mvKeysRight; 原始左图像,右图像提取出的特征点（未校正）
        N = mvKeys.size();
        if(mvKeys.empty()) {
            return;
        }
        //UndistortKeyPoints();
        UndistortLeftKeyPoints();
        UndistortRightKeyPoints();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
        std::chrono::steady_clock::time_point match1 = std::chrono::steady_clock::now();
        undistComputeStereoMatches();
        std::chrono::steady_clock::time_point match2 = std::chrono::steady_clock::now();
        summatchimg ++;
        duration = std::chrono::duration_cast<std::chrono::duration<double> >(match2 - match1).count();
        allduration += duration;
        averduration = allduration / summatchimg;

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();
        mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
        mmMatchedInImage.clear();
        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations) {
            ComputeImageBounds(imLeft);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);
            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;
            mbInitialComputations = false;
        }
        mb = mbf / fx;
        if(pPrevF) {
            if(!pPrevF->mVw.empty()) {
                mVw = pPrevF->mVw.clone();
            }
        } else {
            mVw = cv::Mat::zeros(3, 1, CV_32F);
        }
        AssignFeaturesToGrid();
        mpMutexImu = new std::mutex();
        //Set no stereo fisheye information
        Nleft = -1;
        Nright = -1;
        mvLeftToRightMatch = vector<int>(0);
        mvRightToLeftMatch = vector<int>(0);
        mTlr = cv::Mat(3, 4, CV_32F);
        mTrl = cv::Mat(3, 4, CV_32F);
        mvStereo3Dpoints = vector<cv::Mat>(0);
        monoLeft = -1;
        monoRight = -1;
    }

    /**
     * @brief Frame::CalculMinpoint 计算最近点
     * @param imMask    掩码
     * @param orginpoint　相机的点位置
     * @param minpoint　最近位置
     */
    void Frame::CalculMinpoint(const cv::Mat &imMask, Point2f &orginpoint, Point2f &minpoint) {
        // 变换之前的最小点
        // 直接计算最大的y点
        float maxY = INT_MIN;
        bool flag = false;
        std::unordered_set<int> pixel;
        pixel.clear();
        for(int i = imMask.rows - 1; i >= 0; i--) {
            for(int j = imMask.cols - 1; j >= 0; j--) {
                pixel.insert(imMask.at<unsigned char>(i, j));
                if(imMask.at<unsigned char>(i, j) == 125 || imMask.at<unsigned char>(i, j) == 175) {
                    if(imMask.rows > maxY) {
                        maxY = imMask.rows;
                        minpoint.x = j;
                        minpoint.y = i;

                        flag = true;
                        break;
                    }
                }
            }
            if(flag) {
                break;
            }
        }
        cout << "像素种类 = " << pixel.size() << endl;
        unordered_set<int>::iterator find_iter = pixel.begin();
        while(find_iter != pixel.end()) {
            cout << *find_iter << " ";
            ++find_iter;
        }
        cout << endl;
        cout << "最小距离点坐标=" << minpoint << endl;
    }

    // 计算道路点的位置
    /**
     * @brief Frame::Calculroad3dpointandcalculdistance  计算道路3d点,并计算一个对象的最近距离
     * @param imroadMaskleft    道路掩码
     * @param allpoints     全图与特征点
     * @param mvroadDepth   特征点的深度
     * @param road3dpoint   输出返回的三维点
     */
    void Frame::Calculroad3dpointandcalculdistance(const cv::Mat &imroadMaskleft, std::vector<cv::KeyPoint> allpoints, std::vector<float> mvroadDepth, const cv::Mat &imleft) {

        // 测试读取mask种类
        int maskclassnums = 0;
        Findmaskclass(imroadMaskleft, maskclassnums);

        vector<Point2f> preroad, planeroad;
        preroad.clear();
        planeroad.clear();
        int N = allpoints.size();
        road3dpoint.clear();
//        printMat(LEFTP);
        float cfx = ORB_SLAM3::fx;
        float cfy = ORB_SLAM3::fx;
        float ccx = ORB_SLAM3::Lcx;
        float ccy = ORB_SLAM3::Lcy;
        float sumdistance = 0, averdistance = 0;
        int distancenumber = 0;
        int pixeldistancemultiple = 20;// 距离在像素上放大的倍数

        for(int i = 0; i < N; i++) {
            if(mvroadDepth[i] > 0 && (imroadMaskleft.at<unsigned char>(round(allpoints[i].pt.y), round(allpoints[i].pt.x)) == 255)) {
                float x, y, z;
                z = mvroadDepth[i];
                float u = allpoints[i].pt.x;
                float v = allpoints[i].pt.y;
                x = (u - ccx) * z * (1 / cfx);
                y = (v - ccy) * z * (1 / cfy);
                road3dpoint.emplace_back(Point3f(x, y, z));
                plane2dpoint.emplace_back(Point2f(x * pixeldistancemultiple + imroadMaskleft.cols / 2, imroadMaskleft.rows - z * pixeldistancemultiple));
                // 记录对应的点用于计算H矩阵
                preroad.emplace_back(Point2f(u, v));
                planeroad.emplace_back(Point2f(x * pixeldistancemultiple + imroadMaskleft.cols / 2, imroadMaskleft.rows - z * pixeldistancemultiple));
            } else {
                // 默认一个对象
                // 测试距离是否准确
                if(imroadMaskleft.at<unsigned char>(round(allpoints[i].pt.y), round(allpoints[i].pt.x)) == 125 ||
                        imroadMaskleft.at<unsigned char>(round(allpoints[i].pt.y), round(allpoints[i].pt.x)) == 175) {
                    if(mvroadDepth[i] != -1) {
                        sumdistance += mvroadDepth[i];
                        distancenumber++;
                    }
                }
                road3dpoint.emplace_back(Point3f(0, 0, -1));
                plane2dpoint.emplace_back(Point2f(-1, -1));
            }
        }

        // 使用双目匹配的特征点计算平均距离
        averdistance = sumdistance / distancenumber;
        cout << "双目匹配平均距离 ：" << averdistance << endl;

        // 计算H矩阵
        Mat  H = findHomography(preroad, planeroad, CV_RANSAC);

        // 投射变换
        Mat perspectmaskimg;
        warpPerspective(imroadMaskleft, perspectmaskimg, H, imroadMaskleft.size());

        // 计算最近点的坐标
        Point2f orginpoint(imroadMaskleft.cols / 2, imroadMaskleft.rows);
        // mindistancepoint　一个对象的最近点
        Point2f mindistancepoint;
        // 计算最小距离点
        CalculMinpoint(imroadMaskleft, orginpoint, mindistancepoint);

        // 映射点
        Point2f perspectminpoint;
        TransportpointforH(mindistancepoint, perspectminpoint, H);

        // matchdistance双目匹配计算平均距离
        // perspectdistance 映射计算距离
        matchdistance = averdistance;
        perspectdistance = (orginpoint.y - perspectminpoint.y) / pixeldistancemultiple ;
        cout << "映射最近点坐标 X=" << perspectminpoint.x << "     映射最近点坐标 Y=" << perspectminpoint.y << endl;
        cout << "双目匹配平均距离 :" << averdistance << "m          映射目标距离 :" << perspectdistance << "m" << endl;
        // 测试最小点是否正确

        // tempimroadMaskleft用于画图而不改变原图
        Mat tempimroadMaskleft;
        imroadMaskleft.copyTo(tempimroadMaskleft);
        cv::cvtColor(tempimroadMaskleft, tempimroadMaskleft, cv::COLOR_GRAY2BGR);

        // 在图上画点距离
        Point text0, text;
        text0.x = mindistancepoint.x + 40;
        text0.y = mindistancepoint.y - 40;
        text.x = mindistancepoint.x + 40;
        text.y = mindistancepoint.y + 40;
        putText(tempimroadMaskleft, to_string(averdistance), text0, CV_FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 0, 255));
        putText(tempimroadMaskleft, to_string(perspectdistance), text, CV_FONT_HERSHEY_COMPLEX, 1, Scalar( 255, 0, 255));

        // 测试计算面积和最近点
        int maskvalue = 125;
        int allmaskarea = 0, roimaskarea = 0;
        Point2f outroimaskminpoint, roimaskminpoint;
        // 计算这个掩码值对应的全图面积大小和roi区域面积大小，以及在对应roi内外的最近点图像坐标，但是这一点是根据映射前的mask计算的，但是在映射后这个点不一定是最近点
        // 不计算映射后最小点的原因是因为映射后图像被插值，无法根据图像的mask值找出位置
        Calculmaskarea(imroadMaskleft, maskvalue, allmaskarea, roimaskarea, outroimaskminpoint, roimaskminpoint);
        cout << "所有面积=" << allmaskarea << "    内面积=" << roimaskarea << "   外点=" << outroimaskminpoint << "    内点=" << roimaskminpoint << endl;
        // 仅仅根据面积百分比选择距离，后续可考虑roi内外的点的最近距离
        Choicedistance(averdistance, perspectdistance, allmaskarea, roimaskarea, objbestdistance);

    }

    void Frame::Calculroad3dpointandcalculalldistance(const cv::Mat &imroadMaskleft, std::vector<cv::KeyPoint> allpoints, std::vector<float> mvroadDepth, const cv::Mat &imleft) {

        // 测试读取mask种类
        Findmaskclass(imroadMaskleft, maskclassnums);

        vector<Point2f> preroad, planeroad;
        preroad.clear();
        planeroad.clear();
        int N = allpoints.size();
        road3dpoint.clear();
        float cfx = ORB_SLAM3::fx;
        float cfy = ORB_SLAM3::fx;
        float ccx = ORB_SLAM3::Lcx;
        float ccy = ORB_SLAM3::Lcy;

        int pixeldistancemultiple = 10;// 距离在像素上放大的倍数

        // 自适应生成对象个数的容器，存储各自对象的总距离
        vector<vector<float>> sumdistancemask(maskclassnums);
        // 求去这个mask区域的特征点平均距离
        vector<float> maskaverdistance(maskclassnums);

        for(int i = 0; i < N; i++) {
            if(mvroadDepth[i] > 0 && (imroadMaskleft.at<unsigned char>(round(allpoints[i].pt.y), round(allpoints[i].pt.x)) == 255)) {
                float x, y, z;
                z = mvroadDepth[i];
                float u = allpoints[i].pt.x;
                float v = allpoints[i].pt.y;
                x = (u - ccx) * z * (1 / cfx);
                y = (v - ccy) * z * (1 / cfy);
                road3dpoint.emplace_back(Point3f(x, y, z));
                plane2dpoint.emplace_back(Point2f(x * pixeldistancemultiple + imroadMaskleft.cols / 2, imroadMaskleft.rows - z * pixeldistancemultiple));
                // 记录对应的点用于计算H矩阵
                preroad.emplace_back(Point2f(u, v));
                planeroad.emplace_back(Point2f(x * pixeldistancemultiple + imroadMaskleft.cols / 2, imroadMaskleft.rows - z * pixeldistancemultiple));
            } else {
                // 默认一个对象
                // 测试距离是否准确

                for(int k = 0; k < maskclassnums; k++) { //mask种类数
                    if(imroadMaskleft.at<unsigned char>(round(allpoints[i].pt.y), round(allpoints[i].pt.x)) == (255 - 5 * (k + 1))) {
                        if(mvroadDepth[i] != -1) {
                            sumdistancemask[k].push_back(mvroadDepth[i]);//把mask对应的深度插入到容器
                        }
                    }

                    //　道路外的点，后续可以改进使对象的点插入到对象容器中
                    road3dpoint.emplace_back(Point3f(0, 0, -1));
                    plane2dpoint.emplace_back(Point2f(-1, -1));
                }
            }
        }
        // 使用双目匹配的特征点计算平均距离
        for(int i = 0; i < maskclassnums; i++) {
            float sum = std::accumulate(std::begin(sumdistancemask[i]), std::end(sumdistancemask[i]), 0.0);
            maskaverdistance[i] = sum / sumdistancemask[i].size();
        }


        // 计算H矩阵
        // Mat H, Hlast;已经写入在头文件

        if(preroad.size() > 4) {
            H = findHomography(preroad, planeroad, CV_RANSAC);
            if(H.empty()) {
                Hlast.copyTo(H);
            }

            H.copyTo(Hlast);
        } else if(ORB_SLAM3::squenceimg == 0 && preroad.size() <= 4) {
            preroad.emplace_back(797, 647);
            preroad.emplace_back(464, 760);
            preroad.emplace_back(277, 748);
            preroad.emplace_back(742, 696);
            preroad.emplace_back(507, 776);
            preroad.emplace_back(902, 805);
            preroad.emplace_back(1033, 789);
            preroad.emplace_back(1067, 797);
            preroad.emplace_back(316.92, 665.76);

            planeroad.emplace_back(631.665, 821.488);
            planeroad.emplace_back(603.794, 915.706);
            planeroad.emplace_back(586.498, 877.463);
            planeroad.emplace_back(621.359, 855.534);
            planeroad.emplace_back(606.412, 928.435);
            planeroad.emplace_back(626.646, 890.738);
            planeroad.emplace_back(635.22, 888.207);
            planeroad.emplace_back(619.301, 935.122);
            planeroad.emplace_back(578.825, 832.696);
            H = findHomography(preroad, planeroad, CV_RANSAC);
            // if kitti 测试
            H.at<double>(0, 0) = 0.000265154;
            H.at<double>(0, 1) = -1.45091;
            H.at<double>(0, 2) = 1.20577e+23;
            H.at<double>(1, 0) = 2.90482e-24;
            H.at<double>(1, 1) = -1.13055;
            H.at<double>(1, 2) = -2.79796e+18;
            H.at<double>(2, 0) = -1.54595e+33;
            H.at<double>(2, 1) = -0.521844;
            H.at<double>(2, 2) = 7.52365e+18;

            H.copyTo(Hlast);
        } else if(preroad.size() <= 4) {
            Hlast.copyTo(H);
        }
        // 投射变换
        Mat perspectmaskimg;
        warpPerspective(imroadMaskleft, perspectmaskimg, H, imroadMaskleft.size());

        // 计算最近点的坐标
        Point2f orginpoint(imroadMaskleft.cols / 2, imroadMaskleft.rows);

        // tempimroadMaskleft用于画图而不改变原图
        Mat tempimroadMaskleft;
        imleft.copyTo(tempimroadMaskleft);
        cv::cvtColor(tempimroadMaskleft, tempimroadMaskleft, cv::COLOR_GRAY2BGR);

        maskObjDistance.resize(maskclassnums, vector<float>());
        // 根据mask数量计算所有对象距离
        for(int i = 0; i < maskclassnums; i++) {
            int maskvalue = 250 - i * 5;
            int allmaskarea = 0, roimaskarea = 0;
            Point2f outroimaskminpoint, roimaskminpoint;
            Calculmaskarea(imroadMaskleft, maskvalue, allmaskarea, roimaskarea, outroimaskminpoint, roimaskminpoint);
            Point2f perspectoutroimaskminpoint, perspectroimaskminpoint;
            TransportpointforH(outroimaskminpoint, perspectoutroimaskminpoint, H);
            TransportpointforH(roimaskminpoint, perspectroimaskminpoint, H);
            //选择映射后的最近点
            Point2f betterperspectmaskminpoint, bettermaskminpoint;

            if(outroimaskminpoint.y > roimaskminpoint.y) {
                if(outroimaskminpoint.y != -1) {
                    goodmaskminpoint.push_back(outroimaskminpoint);
                    betterperspectmaskminpoint = perspectoutroimaskminpoint;
                } else {
                    goodmaskminpoint.push_back(roimaskminpoint);
                    betterperspectmaskminpoint = perspectroimaskminpoint;
                }
            } else {
                if(roimaskminpoint.y != -1) {
                    goodmaskminpoint.push_back(roimaskminpoint);
                    betterperspectmaskminpoint = perspectroimaskminpoint;
                } else {
                    goodmaskminpoint.push_back(outroimaskminpoint);
                    betterperspectmaskminpoint = perspectoutroimaskminpoint;
                }
            }
            float preimgdistace = (tempimroadMaskleft.rows - goodmaskminpoint[i].y) / pixeldistancemultiple;
            perspectdistance = (tempimroadMaskleft.rows - betterperspectmaskminpoint.y) / pixeldistancemultiple;
            //　选择最近距离
            Choicebetterdistance(maskaverdistance[i], perspectdistance, preimgdistace, allmaskarea, roimaskarea, objbestdistance);
            maskObjDistance[i].push_back(objbestdistance);//保存每个对象的距离
            // 在图上写出距离
            Point text;
            text.x = goodmaskminpoint[i].x ;
            text.y = goodmaskminpoint[i].y + 12;
        }
    }


    //　计算掩码值对应的在全局的像素面积(数量)和在ROI(右相机视野)的数量,并且返回全图的最近点和roi最近点(未映射点)
    /**
     * @brief Frame::Calculmaskarea
     * @param[in] imroadMaskleft
     * @param[in] maskvalue
     * @param[in&out] allarea   全图的mask个数
     * @param[in&out] roiarea   roi的mask个数
     * @param[in&out] outroiminpoint  roi外的最近点
     * @param[in&out] roiminpoint   roi内的最近点
     */
    void Frame::Calculmaskarea(const cv::Mat &imroadMaskleft, int &maskvalue, int &allarea, int &roiarea, Point2f &outroiminpoint, Point2f &roiminpoint) {
        Point2f initallminpoint(-1, -1);
        Point2f initroiminpoint(-1, -1);
        // 初始为０
        allarea = 0;
        roiarea = 0;
        // i是行，ｊ是列
        // Point LeftUp, RightBottom;这个是最开始识别的ｒｏｉ边界
        for(int i = imroadMaskleft.rows - 1; i >= 0; i--) {
            for(int j = imroadMaskleft.cols - 1; j >= 0; j--) {
                if(imroadMaskleft.at<unsigned char>(i, j) == maskvalue) {
                    allarea++;
                    //判断是否在roi内
                    if((i > ORB_SLAM3::LeftUp.y && i < ORB_SLAM3::RightBottom.y) && (j > ORB_SLAM3::LeftUp.x && j < ORB_SLAM3::RightBottom.x)) {
                        roiarea++;
                        //计算在ｒｏｉ的最小点位置
                        if(i > initroiminpoint.y) {
                            initroiminpoint.x = j;
                            initroiminpoint.y = i;
                        }
                    } else {
                        if(i > initallminpoint.y) {
                            initallminpoint.x = j;
                            initallminpoint.y = i;
                        }
                    }
                }

            }
        }
        outroiminpoint = initallminpoint;
        roiminpoint = initroiminpoint;
    }

    // 计算经过Ｈ矩阵转换的点位置
    /**
     * @brief Frame::TransportpointforH
     * @param inpoint   输入点
     * @param outpoint  输出点
     * @param H         转换矩阵
     */
    void Frame::TransportpointforH(Point2f &inpoint, Point2f &outpoint, Mat &H) {
        cv::Mat_<double> mat_point(3, 1);
        mat_point(0, 0) = inpoint.x;
        mat_point(1, 0) = inpoint.y;
        mat_point(2, 0) = 1;
        Mat mat_tmp = H * mat_point;
        double a1 = mat_tmp.at<double>(0, 0);
        double a2 = mat_tmp.at<double>(1, 0);
        double a3 = mat_tmp.at<double>(2, 0);
        outpoint.x = a1 / a3;
        outpoint.y = a2 / a3;
    }

    // 选择距离函数
    void Frame::Choicedistance(float &visiondistance, float &perspectdistance, int &allarea, int &roiarea, float &bestdistance) {
        if(perspectdistance < 0 && !isnan(visiondistance) ) {
            bestdistance = visiondistance;
        } else if (isnan(visiondistance) && perspectdistance >= 0) {
            bestdistance = perspectdistance;
        } else if (perspectdistance < 0 && isnan(visiondistance)) {
            bestdistance = -1;
        } else {
            float ratio = roiarea / allarea;
            if(ratio > 0.6) {
                bestdistance = visiondistance;
            } else {
                bestdistance = perspectdistance;
            }
        }
    }

    // 选择当映射距离为负数时距离函数
    void Frame::Choicebetterdistance(float &visiondistance, float &perspectdistance, float &preimgdistance, int &allarea, int &roiarea, float &bestdistance) {
        if(perspectdistance < 0 && !isnan(visiondistance) ) {
            bestdistance = visiondistance;
        } else if (isnan(visiondistance) && perspectdistance >= 0) {
            bestdistance = perspectdistance;
        } else if (perspectdistance < 0 && isnan(visiondistance)) {
            bestdistance = preimgdistance;
        } else {
            float ratio = roiarea / allarea;
            if(ratio > 0.6) {
                bestdistance = visiondistance;
            } else {
                bestdistance = perspectdistance;
            }
        }
    }

    //　读取mask非像素值非255和0的种类数
    void Frame::Findmaskclass(const cv::Mat &maskimg, int &classnums) {
        std::unordered_set<int> maskclass;
        for(int i = maskimg.rows - 1; i >= 0; i--) {
            for(int j = maskimg.cols - 1; j >= 0; j--) {
                if(maskimg.at<unsigned char>(i, j) != 0 && maskimg.at<unsigned char>(i, j) != 255) {
                    maskclass.insert(maskimg.at<unsigned char>(i, j));
                }
            }
        }

        classnums = maskclass.size();
    }

    // 输入两个点，调用函数TransportpointforH映射，如果距离为负数，则循环使用相邻点计算，要考虑边界调点
    void Frame::Findtrueperspectdistance(Point2f &inpoint, Point2f &outpoint, Mat &H, const cv::Mat &maskimg, int &times, float &distance) {
        Point2f addoutpoint = outpoint;
        while(addoutpoint.y > maskimg.rows) {
            Point2f addinpoint;
            addinpoint.x = inpoint.x;
            addinpoint.y = max(inpoint.y - 1, (float)0.);
            TransportpointforH(addinpoint, addoutpoint, H);
        }
        distance = (maskimg.rows - addoutpoint.y) / times;
    }

//　stereo with mask.
    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const cv::Mat &imMaskleft, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF, const IMU::Calib &ImuCalib)
        : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false),
          mpCamera(pCamera), mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0) {
        // Frame ID
        mnId = nNextId++;
        maskleft = imMaskleft.clone();
        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        // ORB extraction
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);
        threadLeft.join();
        threadRight.join();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
        mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
        //mvKeys, mvKeysRight; 原始左图像,右图像提取出的特征点（未校正）
        N = mvKeys.size();
        if(mvKeys.empty()) {
            return;
        }

        UndistortLeftKeyPoints();
        UndistortRightKeyPoints();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
        std::chrono::steady_clock::time_point match1 = std::chrono::steady_clock::now();
        undistComputeStereoMatches();
        std::chrono::steady_clock::time_point match2 = std::chrono::steady_clock::now();
        summatchimg ++;
        duration = std::chrono::duration_cast<std::chrono::duration<double> >(match2 - match1).count();
        allduration += duration;
        averduration = allduration / summatchimg;
        // 计算道路掩码分割的3d位置
        /**
         * @brief Calculroad3dpointandcalculdistance 核心函数，通过计算Ｈ矩阵，在计算各个对象的位置，计算一个对象的位置，这个对象mask值为125
         */
    
        Calculroad3dpointandcalculalldistance(imMaskleft, refermvKeys, mvDepthdyna, imLeft);//计算所有对象位置


#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();
        mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
        mmMatchedInImage.clear();
        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations) {
            ComputeImageBounds(imLeft);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);
            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;
            mbInitialComputations = false;
        }
        mb = mbf / fx;
        if(pPrevF) {
            if(!pPrevF->mVw.empty()) {
                mVw = pPrevF->mVw.clone();
            }
        } else {
            mVw = cv::Mat::zeros(3, 1, CV_32F);
        }
        //　放到去除动态对象点后分割特征点到图像中
        AssignFeaturesToGrid();
        mpMutexImu = new std::mutex();
        //Set no stereo fisheye information
        Nleft = -1;
        Nright = -1;
        mvLeftToRightMatch = vector<int>(0);
        mvRightToLeftMatch = vector<int>(0);
        mTlr = cv::Mat(3, 4, CV_32F);
        mTrl = cv::Mat(3, 4, CV_32F);
        mvStereo3Dpoints = vector<cv::Mat>(0);
        monoLeft = -1;
        monoRight = -1;
    }


    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame *pPrevF, const IMU::Calib &ImuCalib)
        : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false),
          mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0) {
        // Frame ID
        mnId = nNextId++;
        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        // ORB extraction
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        ExtractORB(0, imGray, 0, 0);
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
        mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
        N = mvKeys.size();
        if(mvKeys.empty()) {
            return;
        }
        UndistortKeyPoints();
        ComputeStereoFromRGBD(imDepth);
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
        mmMatchedInImage.clear();
        mvbOutlier = vector<bool>(N, false);
        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations) {
            ComputeImageBounds(imGray);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);
            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;
            mbInitialComputations = false;
        }
        mb = mbf / fx;
        AssignFeaturesToGrid();
        mpMutexImu = new std::mutex();
        //Set no stereo fisheye information
        Nleft = -1;
        Nright = -1;
        mvLeftToRightMatch = vector<int>(0);
        mvRightToLeftMatch = vector<int>(0);
        mTlr = cv::Mat(3, 4, CV_32F);
        mTrl = cv::Mat(3, 4, CV_32F);
        mvStereo3Dpoints = vector<cv::Mat>(0);
        monoLeft = -1;
        monoRight = -1;
    }


    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, GeometricCamera *pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame *pPrevF, const IMU::Calib &ImuCalib)
        : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(static_cast<Pinhole *>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
          mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0) {
        // Frame ID
        mnId = nNextId++;
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        // ORB extraction
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        ExtractORB(0, imGray, 0, 1000);
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
        mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
        N = mvKeys.size();
        if(mvKeys.empty()) {
            return;
        }
        UndistortKeyPoints();
        // Set no stereo information
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);
        mnCloseMPs = 0;
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
        mmMatchedInImage.clear();
        mvbOutlier = vector<bool>(N, false);
        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations) {
            ComputeImageBounds(imGray);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);
            fx = static_cast<Pinhole *>(mpCamera)->toK().at<float>(0, 0);
            fy = static_cast<Pinhole *>(mpCamera)->toK().at<float>(1, 1);
            cx = static_cast<Pinhole *>(mpCamera)->toK().at<float>(0, 2);
            cy = static_cast<Pinhole *>(mpCamera)->toK().at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;
            mbInitialComputations = false;
        }
        mb = mbf / fx;
        //Set no stereo fisheye information
        Nleft = -1;
        Nright = -1;
        mvLeftToRightMatch = vector<int>(0);
        mvRightToLeftMatch = vector<int>(0);
        mTlr = cv::Mat(3, 4, CV_32F);
        mTrl = cv::Mat(3, 4, CV_32F);
        mvStereo3Dpoints = vector<cv::Mat>(0);
        monoLeft = -1;
        monoRight = -1;
        AssignFeaturesToGrid();
        if(pPrevF) {
            if(!pPrevF->mVw.empty()) {
                mVw = pPrevF->mVw.clone();
            }
        } else {
            mVw = cv::Mat::zeros(3, 1, CV_32F);
        }
        mpMutexImu = new std::mutex();
    }


    void Frame::AssignFeaturesToGrid() {
        // Fill matrix with points
        const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;
        int nReserve = 0.5f * N / (nCells);
        for(unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
                mGrid[i][j].reserve(nReserve);
                if(Nleft != -1) {
                    mGridRight[i][j].reserve(nReserve);
                }
            }
        for(int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                     : (i < Nleft) ? mvKeys[i]
                                     : mvKeysRight[i - Nleft];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp, nGridPosX, nGridPosY)) {
                if(Nleft == -1 || i < Nleft) {
                    mGrid[nGridPosX][nGridPosY].push_back(i);
                } else {
                    mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
                }
            }
        }
    }

    void Frame::DynaAssignFeaturesToGrid() {
        // Fill matrix with points
        const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;
        int nReserve = 0.5f * N / (nCells);
        mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS].clear();
        for(unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
                mGrid[i][j].reserve(nReserve);
                if(Nleft != -1) {
                    mGridRight[i][j].reserve(nReserve);
                }
            }
        for(int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                     : (i < Nleft) ? mvKeys[i]
                                     : mvKeysRight[i - Nleft];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp, nGridPosX, nGridPosY)) {
                if(Nleft == -1 || i < Nleft) {
                    mGrid[nGridPosX][nGridPosY].push_back(i);
                } else {
                    mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
                }
            }
        }
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1) {
        vector<int> vLapping = {x0, x1};
        if(flag == 0) {
            monoLeft = (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping, flag);
        } else {
            monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping, flag);
        }
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::GetPose(cv::Mat &Tcw) {
        Tcw = mTcw.clone();
    }

    void Frame::SetNewBias(const IMU::Bias &b) {
        mImuBias = b;
        if(mpImuPreintegrated) {
            mpImuPreintegrated->SetNewBias(b);
        }
    }

    void Frame::SetVelocity(const cv::Mat &Vwb) {
        mVw = Vwb.clone();
    }

    void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb) {
        mVw = Vwb.clone();
        cv::Mat Rbw = Rwb.t();
        cv::Mat tbw = -Rbw * twb;
        cv::Mat Tbw = cv::Mat::eye(4, 4, CV_32F);
        Rbw.copyTo(Tbw.rowRange(0, 3).colRange(0, 3));
        tbw.copyTo(Tbw.rowRange(0, 3).col(3));
        mTcw = mImuCalib.Tcb * Tbw;
        UpdatePoseMatrices();
    }



    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    cv::Mat Frame::GetImuPosition() {
        return mRwc * mImuCalib.Tcb.rowRange(0, 3).col(3) + mOw;
    }

    cv::Mat Frame::GetImuRotation() {
        return mRwc * mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3);
    }

    cv::Mat Frame::GetImuPose() {
        cv::Mat Twb = cv::Mat::eye(4, 4, CV_32F);
        Twb.rowRange(0, 3).colRange(0, 3) = mRwc * mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3);
        Twb.rowRange(0, 3).col(3) = mRwc * mImuCalib.Tcb.rowRange(0, 3).col(3) + mOw;
        return Twb.clone();
    }


    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        if(Nleft == -1) {
            pMP->mbTrackInView = false;
            pMP->mTrackProjX = -1;
            pMP->mTrackProjY = -1;
            // 3D in absolute coordinates
            cv::Mat P = pMP->GetWorldPos();
            // 3D in camera coordinates
            const cv::Mat Pc = mRcw * P + mtcw;
            const float Pc_dist = cv::norm(Pc);
            // Check positive depth
            const float &PcZ = Pc.at<float>(2);
            const float invz = 1.0f / PcZ;
            if(PcZ < 0.0f) {
                return false;
            }
            const cv::Point2f uv = mpCamera->project(Pc);
            if(uv.x < mnMinX || uv.x > mnMaxX) {
                return false;
            }
            if(uv.y < mnMinY || uv.y > mnMaxY) {
                return false;
            }
            pMP->mTrackProjX = uv.x;
            pMP->mTrackProjY = uv.y;
            // Check distance is in the scale invariance region of the MapPoint
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const cv::Mat PO = P - mOw;
            const float dist = cv::norm(PO);
            if(dist < minDistance || dist > maxDistance) {
                return false;
            }
            // Check viewing angle
            cv::Mat Pn = pMP->GetNormal();
            const float viewCos = PO.dot(Pn) / dist;
            if(viewCos < viewingCosLimit) {
                return false;
            }
            // Predict scale in the image
            const int nPredictedLevel = pMP->PredictScale(dist, this);
            // Data used by the tracking
            pMP->mbTrackInView = true;
            pMP->mTrackProjX = uv.x;
            pMP->mTrackProjXR = uv.x - mbf * invz;
            pMP->mTrackDepth = Pc_dist;
            pMP->mTrackProjY = uv.y;
            pMP->mnTrackScaleLevel = nPredictedLevel;
            pMP->mTrackViewCos = viewCos;
            return true;
        } else {
            pMP->mbTrackInView = false;
            pMP->mbTrackInViewR = false;
            pMP -> mnTrackScaleLevel = -1;
            pMP -> mnTrackScaleLevelR = -1;
            pMP->mbTrackInView = isInFrustumChecks(pMP, viewingCosLimit);
            pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, true);
            return pMP->mbTrackInView || pMP->mbTrackInViewR;
        }
    }

    bool Frame::ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v) {
        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();
        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);
        // Check positive depth
        if(PcZ < 0.0f) {
            cout << "Negative depth: " << PcZ << endl;
            return false;
        }
        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        u = fx * PcX * invz + cx;
        v = fy * PcY * invz + cy;
        if(u < mnMinX || u > mnMaxX) {
            return false;
        }
        if(v < mnMinY || v > mnMaxY) {
            return false;
        }
        float u_distort, v_distort;
        float x = (u - cx) * invfx;
        float y = (v - cy) * invfy;
        float r2 = x * x + y * y;
        float k1 = mDistCoef.at<float>(0);
        float k2 = mDistCoef.at<float>(1);
        float p1 = mDistCoef.at<float>(2);
        float p2 = mDistCoef.at<float>(3);
        float k3 = 0;
        if(mDistCoef.total() == 5) {
            k3 = mDistCoef.at<float>(4);
        }
        // Radial distorsion
        float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
        float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
        // Tangential distorsion
        x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
        y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);
        u_distort = x_distort * fx + cx;
        v_distort = y_distort * fy + cy;
        u = u_distort;
        v = v_distort;
        kp = cv::Point2f(u, v);
        return true;
    }

    cv::Mat Frame::inRefCoordinates(cv::Mat pCw) {
        return mRcw * pCw + mtcw;
    }

// 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float   &y, const float   &r, const int minLevel, const int maxLevel, const bool bRight) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);
        float factorX = r;
        float factorY = r;
        const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
        if(nMinCellX >= FRAME_GRID_COLS) {
            return vIndices;
        }
        const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
        if(nMaxCellX < 0) {
            return vIndices;
        }
        const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
        if(nMinCellY >= FRAME_GRID_ROWS) {
            return vIndices;
        }
        const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
        if(nMaxCellY < 0) {
            return vIndices;
        }
        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);
        for(int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for(int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
                if(vCell.empty()) {
                    continue;
                }
                for(size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]] : (!bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                    if(bCheckLevels) {
                        if(kpUn.octave < minLevel) {
                            continue;
                        }
                        if(maxLevel >= 0)
                            if(kpUn.octave > maxLevel) {
                                continue;
                            }
                    }
                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;
                    if(fabs(distx) < factorX && fabs(disty) < factorY) {
                        vIndices.push_back(vCell[j]);
                    }
                }
            }
        }
        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);
        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS) {
            return false;
        }
        return true;
    }


    void Frame::ComputeBoW() {
        if(mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if(mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }
        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for(int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);
        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for(int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }



    void Frame::UndistortLeftKeyPoints() {
        // K1 ,K2, P1,P2
        if(LEFTD.at<float>(0) == 0.0 && ORB_SLAM3::fx == ORB_SLAM3::Rfx) {
            mvKeysUn = mvKeys;
            refermvKeys = mvKeys;
            mvKeysdyna = mvKeys;
            mvKeysUndyna = mvKeysUn;
            refermvKeysdyna = refermvKeys;
            return;
        }
        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for(int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, ORB_SLAM3::LEFTK, ORB_SLAM3::LEFTD, ORB_SLAM3::LEFTR, ORB_SLAM3::LEFTP);
        mat = mat.reshape(1);
        mvKeysUn.resize(N);
        refermvKeys.clear();
        refermvKeys.resize(N);
        refermvKeys = mvKeys;
        for(int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            if(0 > mat.at<float>(i, 0) || mat.at<float>(i, 0) > width - 1) {
                continue;
            }
            if(0 > mat.at<float>(i, 1) || mat.at<float>(i, 1) > height - 1) {
                continue;
            }
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeys[i] = kp;
            mvKeysUn[i] = kp;
        }
        mvKeysdyna = mvKeys;
        mvKeysUndyna = mvKeysUn;
        refermvKeysdyna = refermvKeys;
    }

    void Frame::UndistortRightKeyPoints() {
        // K1 ,K2, P1,P2
        if(RIGHTD.at<float>(0) == 0.0  && ORB_SLAM3::fx == ORB_SLAM3::Rfx) {
            mvKeysRightUn = mvKeysRight;
            refermvKeysRight = mvKeysRight;
            mvKeysRightdyna = mvKeysRight;
            mvKeysRightUndyna = mvKeysRightUn;
            refermvKeysRightdyna = refermvKeysRight;
            return;
        }
        const int Nr = mvKeysRight.size();
        // Fill matrix with points
        cv::Mat mat(Nr, 2, CV_32F);
        for(int i = 0; i < Nr; i++) {
            mat.at<float>(i, 0) = mvKeysRight[i].pt.x;
            mat.at<float>(i, 1) = mvKeysRight[i].pt.y;
        }
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, ORB_SLAM3::RIGHTK, ORB_SLAM3::RIGHTD, ORB_SLAM3::RIGHTR, ORB_SLAM3::RIGHTP);
        mat = mat.reshape(1);
        mvKeysRightUn.resize(Nr);
        refermvKeysRight.clear();
        refermvKeysRight.resize(Nr);
        refermvKeysRight = mvKeysRight;
        for(int i = 0; i < Nr; i++) {
            cv::KeyPoint kp = mvKeysRight[i];
            if(0 > mat.at<float>(i, 0) || mat.at<float>(i, 0) > width - 1) {
                continue;
            }
            if(0 > mat.at<float>(i, 1) || mat.at<float>(i, 1) > height - 1) {
                continue;
            }
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysRight[i] = kp;
            mvKeysRightUn[i] = kp;
        }
        mvKeysRightdyna = mvKeysRight;
        mvKeysRightUndyna = mvKeysRightUn;
        refermvKeysRightdyna = refermvKeysRight;
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if(mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);
            // Undistort corners
            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void PrintmvuRight(vector<float> &print) {
        int num = print.size();
        int matchedpoint = 0;
        int j = 20;
        for(int i = 0; i < num; i++) {
            if(j > 0) {
                if(print[i] != -1) {
                    matchedpoint++;
                }
                j--;
            } else {
                j = 20;
                if(print[i] != -1) {
                    matchedpoint++;
                }
            }
        }
        cout << "Number of KeyPoints= " << num << endl;
        cout << "Matched in  KeyPoints= " << matchedpoint << endl;
        cout << "-----------------------------------------" << endl;
        cout << endl;
    }

    void PrintmvuLeft(vector<pair<int, int> > &printsecond) {
        int num = printsecond.size();
        int matchedpoint = 0;
        int j = 20;
        for(int i = 0; i < num; i++) {
            if(j > 0) {
                if(printsecond[i].second != -1) {
                    matchedpoint++;
                }
                cout << printsecond[i].second << '\t';
                j--;
            } else {
                j = 20;
                if(printsecond[i].second != -1) {
                    matchedpoint++;
                }
                cout << printsecond[i].second << '\t' ;
                cout << endl;
            }
        }
        cout << "Number of left KeyPoints= " << num << endl;
        cout << "Matched in left KeyPoints= " << matchedpoint << endl;
        cout << "-----------------------------------------" << endl;
        cout << endl;
    }


    /* ************** 原匹配方法***************************** */
    void Frame::ComputeStereoMatches() {
        mvuRight = vector<float>(N, -1.0f);
        mvuLeft = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);
        LeftIdtoRightId = vector<float>(N, -1.0f); //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        RightIdToLeftId = vector<float>(mvKeysRight.size(), -1.0f);
        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;
        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
        //Assign keypoints to row table
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());
        for(int i = 0; i < nRows; i++) {
            vRowIndices[i].reserve(200);
        }
        const int Nr = mvKeysRight.size();
        for(int iR = 0; iR < Nr; iR++) {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);
            for(int yi = minr; yi <= maxr; yi++) {
                vRowIndices[yi].push_back(iR);
            }
        }
        // Set limits for search
        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf / minZ;
        // For each left keypoint search a match in the right image
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);
        int matchedLeftnums = 0;
        for(int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;
            const vector<size_t> &vCandidates = vRowIndices[vL];
            if(vCandidates.empty()) {
                continue;
            }
            const float minU = uL - maxD;
            const float maxU = uL - minD;
            if(maxU < 0) {
                continue;
            }
            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;
            const cv::Mat &dL = mDescriptors.row(iL);
            // Compare descriptor to right keypoints
            for(size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];
                if(kpR.octave < levelL - 1 || kpR.octave > levelL + 1) {
                    continue;
                }
                const float &uR = kpR.pt.x;
                if(uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);
                    if(dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }
            // Subpixel match by correlation
            if(bestDist < thOrbDist) {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);
                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
                //IL.convertTo(IL,CV_32F);
                //IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
                IL.convertTo(IL, CV_16S);
                IL = IL - IL.at<short>(w, w);
                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);
                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if(iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols) {
                    continue;
                }
                for(int incR = -L; incR <= +L; incR++) {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    //IR.convertTo(IR,CV_32F);
                    //IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                    IR.convertTo(IR, CV_16S);
                    IR = IR - IR.at<short>(w, w);
                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if(dist < bestDist) {
                        bestDist =  dist;
                        bestincR = incR;
                    }
                    vDists[L + incR] = dist;
                }
                if(bestincR == -L || bestincR == L) {
                    continue;
                }
                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];
                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));
                if(deltaR < -1 || deltaR > 1) {
                    continue;
                }
                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);
                float disparity = (uL - bestuR);
                if(disparity >= minD && disparity < maxD) {
                    if(disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                    mvuLeft[iL] = iL;
                    matchedLeftnums++;
                    LeftIdtoRightId[iL] = bestIdxR; //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
                    RightIdToLeftId[bestIdxR] = bestIdxR; // 右特征点ｉｄ
                }
            }
        }
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;
        for(int i = vDistIdx.size() - 1; i >= 0; i--) {
            if(vDistIdx[i].first < thDist) {
               break;
            } else {
                matchedLeftnums--;
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
                mvuLeft[vDistIdx[i].second] = -1;
                LeftIdtoRightId[vDistIdx[i].second] = -1; //删除左图离缺点,标志为-1
                RightIdToLeftId[LeftIdtoRightId[vDistIdx[i].second]] = -1;
            }
        }
        cout << endl;
        cout << "matchedLeftnums= " << matchedLeftnums << endl;
    }

    /* 使用去畸变的点匹配特征点,在sad用原特征点滑块匹配，在去畸变点计算深度　*/
    void Frame::undistComputeStereoMatches() {

        // 原slam特征匹配记录点匹配信息
        mvuRight = vector<float>(N, -1.0f); //mvuRight存储右图匹配点索引
        mvuLeft = vector<float>(N, -1.0f);  //mvuLeft存储左图匹配点索引
        mvDepth = vector<float>(N, -1.0f);  //mvDepth存储特征点的深度信息
        LeftIdtoRightId = vector<float>(N, -1.0f); //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        RightIdToLeftId = vector<float>(mvKeysRight.size(), -1.0f);

        mvuRightdyna = vector<float>(N, -1.0f); //mvuRight存储右图匹配点索引
        mvuLeftdyna = vector<float>(N, -1.0f);  //mvuLeft存储左图匹配点索引
        mvDepthdyna = vector<float>(N, -1.0f);  //mvDepth存储特征点的深度信息
        LeftIdtoRightIddyna = vector<float>(N, -1.0f); //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        RightIdToLeftIddyna = vector<float>(mvKeysRight.size(), -1.0f);

        // dyam_slam将对原容器进行复制值，在对原容器是动态点的对象进行点的去除，用复制后特征容器对各项匹配和可视化等等操作
        // 其目的就是剔除原来容器中是动态对象的点，并且还能记录动态点和全图的匹配关系，这一部分将在tracking线程中完成

        // orb特征相似度阈值  -> mean ～= (max  + min) / 2 =75;
        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;
        // 金字塔顶层（0层）图像高 nRows
        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
        // 二维vector存储每一行的orb特征点的列坐标，为什么是vector，因为每一行的特征点有可能不一样，例如
        // vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
        // vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());
        for(int i = 0; i < nRows; i++) {
            vRowIndices[i].reserve(400);
        }
        // 右图特征点数量，N表示数量 r表示右图，且不能被修改
        const int Nr = mvKeysRight.size();
        // Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
        for(int iR = 0; iR < Nr; iR++) {
            // 获取特征点ir的y坐标，即行号
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            // 2.0 与6.0 不影响匹配时间
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = min( ceil(kpY + r),  height - 1.f);
            const int minr = max( floor(kpY - r), 0.f);
              // 将特征点ir保证在可能的行号中
            for(int yi = minr; yi <= maxr; yi++) {
                vRowIndices[yi].push_back(iR);
            }
        }
        // need improve   const float maxD=fx/Fscale;
        const float minZ = mb;//minZ深度
        const float minD = 0;
        const float maxD = mbf / minZ; //mbf:相机的基线长度 * 相机的焦距
        // 保存sad块匹配相似度和左图特征点索引
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);
        // 统计特征数
        int matchedLeftnums = 0;
        int firstlevelpoints = 0;
        for(int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];//变换后的点
            const cv::KeyPoint &refkpL = refermvKeys[iL];//原来的点
            const int &levelL = kpL.octave;
            // 特征点行号
            const float &vL = kpL.pt.y;
            //　行搜索范围
//        if(vL < LeftUp.y || vL > RightBottom.y) continue;
            const float &uL = kpL.pt.x;  //特征点列号
            // 列搜索范围
            //　后续同样使用NCC找边界条件限制匹配
            if(uL < LeftUp.x || uL > RightBottom.x  ) {
                continue;
            }
            const vector<size_t> &vCandidates = vRowIndices[vL];
            if(vCandidates.empty()) {
                continue;
            }
            // 计算理论上的最佳搜索范围
            float minU = uL - maxD; //搜索最小列
            minU = max( minU, 0.0f);
            float maxU = uL - minD ;
            maxU = min( maxU, width - 1.0f);
            // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
            int bestDist = ORBmatcher::TH_HIGH;//100;
            // 记录有特征点序号，即左特征点序列号和又特征点序列号的匹配
            size_t bestIdxR = -1;
            //(il:作图特征点编号)左目摄像头和右目摄像头特征点对应的描述子 mDescriptors, mDescriptorsRight;
            const cv::Mat &dL = mDescriptors.row(iL);//dL用来计算描述子的汉明距离；但描述子的row表示什么？
            // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
            for(size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                cv::KeyPoint &kpR = mvKeysRight[iR];
                /* ****尺度信息不在适用，理论上要金字塔层*****/
                // 右图占左图的0.75，约为1/1.15^2,设置缩放尺度为1.15,假设左图第０层，右图则在第２层
                // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃;
                // Leyermis 长短焦距相差层数  <图像金字塔的尺度因子的对数值 mfLogScaleFactor = log(mfScaleFactor);
                int Leyermis = round(log(Fscale) / mfLogScaleFactor);
                if(kpR.octave - Leyermis < levelL - 1 || kpR.octave - Leyermis > levelL + 1) {
                    continue;
                }
                // 使用列坐标(x)进行匹配，和stereomatch一样,
                const float &uR = kpR.pt.x;
                if(uR >= minU && uR <= maxU) {
                    // 计算匹配点il和待匹配点ic的相似度dist
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);
                    //统计最小相似度及其对应的列坐标(x)
                    if( dist < bestDist ) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }
            // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
            // Step 3. 精确匹配.
            // const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2=75;
            if(bestDist < thOrbDist && bestDist >= 0) {
                // 计算右图特征点x坐标和对应的金字塔尺度
                const float uR0 = refermvKeysRight[bestIdxR].pt.x;
                const float vR0 = refermvKeysRight[bestIdxR].pt.y;
                //scaleFactor=0.83^n,理论上ｎ>=2
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                // +2? 匹配右图的特征图层是要比左图高两层的
                const cv::KeyPoint &kpr = mvKeysRight[bestIdxR];
                const float scaleFactorright = mvInvScaleFactors[kpr.octave];
                const float scaleduL = round(refkpL.pt.x * scaleFactor); //变换前的点
                const float scaledvL = round(refkpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactorright);
                const float scaledvR0 = round(vR0 * scaleFactorright);
                // 滑动窗口搜索, 类似模版卷积或滤波
                // w表示sad相似度的窗口半径
                const int w = 5;
                float leftsadrowmin = 0.;
                float leftsadrowmax = mpORBextractorLeft->mvImagePyramid[kpL.octave].rows;
                float leftsadcolmin = 0.;
                float leftsadcolmax = mpORBextractorLeft->mvImagePyramid[kpL.octave].cols;
                if(scaledvL - w < 0 || scaledvL + w + 1 > leftsadrowmax || scaleduL - w < 0 || scaleduL + w + 1 > leftsadcolmax) {
                    continue;
                }
                // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
              
                IL.convertTo(IL, CV_16S);
                IL = IL - IL.at<short>(w, w);
                //初始化最佳相似度
                int bestDist = INT_MAX;
                // 通过滑动窗口搜索优化，得到的列坐标偏移量
                int bestincR = 0;
                //滑动窗口的滑动范围为（-L, L）
                const int L = 5;
                // 初始化存储图像块相似度
                vector<float> vDists;
                vDists.resize(2 * L + 1);
                // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
                // 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
                // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
                // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
                const float iniu = scaleduR0 + L - w; // scaleduR0:右图粗匹配到的金字塔尺度的特征点坐标ｘ
                const float endu = scaleduR0 + L + w + 1;
                // 判断搜索是否越界
                //  转换后怎么判断
                const cv::KeyPoint &refkpR = refermvKeysRight[bestIdxR];
                const cv::KeyPoint &kpR = mvKeysRight[bestIdxR];
                if(iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpR.octave].cols) {
                    continue;
                }
                // 在搜索范围内从左到右滑动，并计算图像块相似度
                for(int incR = -L; incR <= +L; incR++) {
                    // 提取右图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                    //因为是特征匹配之后的，所以对应的快匹配也是在尺度缩放后的图层上，因此滑块计算的区域是一样的场景
                    float sadrowmin = 0.;
                    float sadrowmax = mpORBextractorRight->mvImagePyramid[kpR.octave].rows;
                    float sadcolmin = 0.;
                    float sadcolmax = mpORBextractorRight->mvImagePyramid[kpR.octave].cols;
                    if(scaledvR0 - w < 0 || scaledvR0 + w + 1 > sadrowmax || scaleduR0 + incR - w < 0 || scaleduR0 + incR + w + 1 > sadcolmax ) {
                        continue;
                    }
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpR.octave].rowRange(scaledvR0 - w, scaledvR0 + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    //cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpR.octave].rowRange(sadrowmin,sadrowmax).colRange(sadcolmin,sadcolmax);
                    //IR.convertTo(IR,CV_32F);
                    // 图像块均值归一化，降低亮度变化对相似度计算的影响
                    //IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                    // slam3用的是１６位。slam2用的３２位；可能是为了加快速度
                    IR.convertTo(IR, CV_16S);
                    IR = IR - IR.at<short>(w, w);
                    // sad 计算
                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    // 统计最小sad和偏移量
                    if(dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }
                    //L+incR 为refine后的匹配点列坐标(x)
                    vDists[L + incR] = dist;
                }
                // 搜索窗口越界判断ß
                // 其实bestincR＝incR，因此这一条只是判断知否在边缘上
                if(bestincR == -L || bestincR == L) {
                    continue;
                }
                // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
                // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
                //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
                //      .             .(16)
                //         .14     .(15) <- int/uchar最佳视差值
                //              .
                //           （14.5）<- 真实的视差值
                //   deltaR = 15.5 - 16 = -0.5
                // 公式参考opencv sgbm源码中的亚像素插值公式
                // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
                const float dist1 = vDists[L + bestincR - 1];	//bestincR:列坐标偏移量
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];
                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));
                // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
                if(deltaR < -1 || deltaR > 1) {
                    continue;
                }
                // 根据亚像素精度偏移量delta调整最佳匹配索引
                //? 为什么用左图的图像尺度？　因为在视差计算时，是根据左图的uL（如下disparity = (uL-bestuR)）得来的
                float bestuR = mvScaleFactors[kpR.octave] * ((float)scaleduR0 + (float)bestincR + deltaR) ;
                //　变换参考点
                //Point bestuR1(bestuR , vR0);
                cv::Mat mat(1, 2, CV_32F);
                mat.at<float>(0, 0) = bestuR;
                mat.at<float>(0, 1) = vR0;
                mat = mat.reshape(2);
                cv::undistortPoints(mat, mat, ORB_SLAM3::RIGHTK, ORB_SLAM3::RIGHTD, ORB_SLAM3::RIGHTR, ORB_SLAM3::RIGHTP);
                mat = mat.reshape(1);
                bestuR = mat.at<float>(0, 0);
                //vR0 = mat.at<float>(0,1);
                // disparity:求得的视差值
                //bestuR = bestuR * invFscale + Lcx - invFscale * Rcx ;
                //bestuR = bestuR * 0.75973+ Lcx - 0.97975 * Rcx ;
                float disparity = uL - bestuR;
                if(disparity >= 0 && disparity < maxD) {
                    // 无穷远
                    if(disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                    mvuLeft[iL] = iL;
                    matchedLeftnums++;
                    if(mvKeysRight[bestIdxR].octave == 0) {
                        firstlevelpoints++;
                    }
                    LeftIdtoRightId[iL] = bestIdxR; // 左特征点对应右特征点ID,IL位置维IR
                    RightIdToLeftId[bestIdxR] = bestIdxR; // 右特征点ID
                }
            }
        }//遍历左图每个特征点
        // Step 6. 删除离缺点(outliers)
        // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
        // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4 * median;
        for(int i = vDistIdx.size() - 1; i >= 0; i--) {
            if(vDistIdx[i].first < thDist) {
                break;
            } else {
                matchedLeftnums--;
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
                mvuLeft[vDistIdx[i].second] = -1;
                RightIdToLeftId[LeftIdtoRightId[vDistIdx[i].second]] = -1;
                LeftIdtoRightId[vDistIdx[i].second] = -1; //删除左图离缺点,标志为-1
            }
        }
        matchednums = 0 ;
        matchednums = matchedLeftnums;
        // 深拷贝相关匹配的对象容器,用于去除动态对象点 20210309 用于dyna_slam
        // 先清空

        mvuRightdyna = mvuRight;
        mvuLeftdyna = mvuLeft;
        mvDepthdyna = mvDepth;
        //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        LeftIdtoRightIddyna = LeftIdtoRightId;
        RightIdToLeftIddyna = RightIdToLeftId;
    }


    /* ************** 长段焦距匹配方法*********************** */
    void Frame::ComputeDifferentfocalStereoMatches() {
        mvuRight = vector<float>(N, -1.0f); //mvuRight存储右图匹配点索引
        mvuLeft = vector<float>(N, -1.0f);  //mvuLeft存储左图匹配点索引
        mvDepth = vector<float>(N, -1.0f);  //mvDepth存储特征点的深度信息
        LeftIdtoRightId = vector<float>(N, -1.0f); //左特征点对应右特征点ｉｄ,ｉＬ位置维Ｉr
        RightIdToLeftId = vector<float>(mvKeysRight.size(), -1.0f);
        // orb特征相似度阈值  -> mean ～= (max  + min) / 2 =75;
        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;
        // 金字塔顶层（0层）图像高 nRows
        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
        // 二维vector存储每一行的orb特征点的列坐标，为什么是vector，因为每一行的特征点有可能不一样，例如
        // vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
        // vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());
        for(int i = 0; i < nRows; i++) {
            vRowIndices[i].reserve(200);
        }
        // 右图特征点数量，N表示数量 r表示右图，且不能被修改
        const int Nr = mvKeysRight.size();
        // Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
        for(int iR = 0; iR < Nr; iR++) {
            // 获取特征点ir的y坐标，即行号
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
            // 2 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = min( ceil(kpY + r), (float) height - 1);
            const int minr = max( floor(kpY - r), 1.f);
            // 将特征点ir保证在可能的行号中
            for(int yi = minr; yi <= maxr; yi++) {
                vRowIndices[yi].push_back(iR);
            }
        }
        // need improve   const float maxD=fx/Fscale;
        const float minZ = mb;//minZ深度
        const float minD = 0;
        const float maxD = mbf / minZ; //mbf:相机的基线长度 * 相机的焦距
        // 保存sad块匹配相似度和左图特征点索引
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);
        // 统计特征数
        int matchedLeftnums = 0;
        //　限制行匹配的范围
        float leftcolmin = LeftUp.x;
        float leftrowmin = LeftUp.y;
        float leftcolmax = RightBottom.x;
        float leftrowmax = RightBottom.y;
        for(int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            // 特征点行号
            const float &vL = kpL.pt.y;
            //　行搜索范围
            if(vL < leftrowmin || vL > leftrowmax ) {
                continue;
            }
            const float &uL = kpL.pt.x;  //特征点列号
            // 列搜索范围
            if(uL < leftcolmin || uL > leftcolmax) {
                continue;
            }
            // 获取左图特征点il所在行，以及在右图映射行中可能的匹配点，映射关系
            if (2 > (vL - leftrowmin)*Fscale || (vL - leftrowmin)*Fscale > (height - 2.0f)) {
                continue;
            }
            const vector<size_t> &vCandidates = vRowIndices[(size_t)(vL - leftrowmin) * Fscale];
            if(vCandidates.empty()) {
                continue;
            }
            // 计算理论上的最佳搜索范围
            float minU = (uL - leftcolmin) * Fscale - maxD; //搜索最小列
            minU = max( minU, 1.0f);
            float maxU = (uL - leftcolmin) * Fscale * 1.2 ; //搜索最大列,乘1.2因为之前感性区域的列范围不够精准！
            maxU = min( maxU, width - 1.0f);
            // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
            int bestDist = ORBmatcher::TH_HIGH;//100;
            size_t bestIdxR = -1;
            //(il:作图特征点编号)左目摄像头和右目摄像头特征点对应的描述子 mDescriptors, mDescriptorsRight;
            const cv::Mat &dL = mDescriptors.row(iL);//dL用来计算描述子的汉明距离；但描述子的row表示什么？
            // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
            for(size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                cv::KeyPoint &kpR = mvKeysRight[iR];
                /* ****尺度信息不在适用，理论上要金字塔层*****/
                // 右图占左图的0.75，约为1/1.15^2,设置缩放尺度为1.15,假设左图第０层，右图则在第２层
                // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃;
                // Leyermis 长短焦距相差层数  <图像金字塔的尺度因子的对数值 mfLogScaleFactor = log(mfScaleFactor);
                int Leyermis = round(log(Fscale) / mfLogScaleFactor);
                if(kpR.octave - Leyermis < levelL - 1 || kpR.octave - Leyermis > levelL + 1) {
                    continue;
                }
                // 使用列坐标(x)进行匹配，和stereomatch一样,
                const float &uR = kpR.pt.x;
                if(uR >= minU && uR <= maxU) {
                    // 计算匹配点il和待匹配点ic的相似度dist
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);
                    //统计最小相似度及其对应的列坐标(x)
                    if( dist < bestDist ) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }
            // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
            // Step 3. 精确匹配.
            // const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2=75;
            if(bestDist < thOrbDist && bestDist >= 0) {
                // 计算右图特征点x坐标和对应的金字塔尺度
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float vR0 = mvKeysRight[bestIdxR].pt.y;
                //scaleFactor=0.83^n,理论上ｎ>=2
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                // +2? 匹配右图的特征图层是要比左图高两层的
                const cv::KeyPoint &kpr = mvKeysRight[bestIdxR];
                const float scaleFactorright = mvInvScaleFactors[kpr.octave];//? 尺度问题
                // 尺度缩放后的左右图特征点坐标
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactorright);
                const float scaledvR0 = round(vR0 * scaleFactorright);
                // 滑动窗口搜索, 类似模版卷积或滤波
                // w表示sad相似度的窗口半径
                const int w = 5;
                // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
                // convertTo()函数负责转换数据类型不同的Mat，即可以将类似float型的Mat转换到imwrite()函数能够接受的类型。
                // 图像块均值归一化，降低亮度变化对相似度计算的影响
                IL.convertTo(IL, CV_16S);
                IL = IL - IL.at<short>(w, w);
                //初始化最佳相似度
                int bestDist = INT_MAX;
                // 通过滑动窗口搜索优化，得到的列坐标偏移量
                int bestincR = 0;
                //滑动窗口的滑动范围为（-L, L）
                const int L = 5;
                // 初始化存储图像块相似度
                vector<float> vDists;
                vDists.resize(2 * L + 1);
                // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
                // 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
                // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
                // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
                const float iniu = scaleduR0 + L - w; // scaleduR0:右图粗匹配到的金字塔尺度的特征点坐标ｘ
                const float endu = scaleduR0 + L + w + 1;
                // 判断搜索是否越界
                //  转换后怎么判断
                const cv::KeyPoint &kpR = mvKeysRight[bestIdxR];
                if(iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpR.octave].cols) {
                    continue;
                }
                // 在搜索范围内从左到右滑动，并计算图像块相似度
                for(int incR = -L; incR <= +L; incR++) {
                    // 提取右图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                    //因为是特征匹配之后的，所以对应的快匹配也是在尺度缩放后的图层上，因此滑块计算的区域是一样的场景
                    float sadrowmin = 0;
                    float sadrowmax = mpORBextractorRight->mvImagePyramid[kpR.octave].rows;
                    float sadcolmin = 0;
                    float sadcolmax = mpORBextractorRight->mvImagePyramid[kpR.octave].cols;
                    sadrowmin = max(sadrowmin, scaledvR0 - w);
                    sadrowmax = min(sadrowmax, scaledvR0 + w + 1);
                    sadcolmin = max(sadcolmin, scaleduR0 + incR - w);
                    sadcolmax = min(sadcolmax, scaleduR0 + incR + w + 1);
                    //? 尺度问题
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpR.octave].rowRange(sadrowmin, sadrowmax).colRange(sadcolmin, sadcolmax);
                    // 图像块均值归一化，降低亮度变化对相似度计算的影响
                    // slam3用的是１６位。ｓｌａｍ2用的３２位；可能是为了加快速度
                    IR.convertTo(IR, CV_16S);
                    IR = IR - IR.at<short>(w, w);
                    // sad 计算
                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    // 统计最小sad和偏移量
                    if(dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }
                    //L+incR 为refine后的匹配点列坐标(x)
                    vDists[L + incR] = dist;
                }
                // 搜索窗口越界判断ß
                // 其实bestincR＝incR，因此这一条只是判断知否在边缘上
                if(bestincR == -L || bestincR == L) {
                    continue;
                }
                // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
                // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
                //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
                //      .             .(16)
                //         .14     .(15) <- int/uchar最佳视差值
                //              .
                //           （14.5）<- 真实的视差值
                //   deltaR = 15.5 - 16 = -0.5
                // 公式参考opencv sgbm源码中的亚像素插值公式
                // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
                const float dist1 = vDists[L + bestincR - 1];	//bestincR:列坐标偏移量
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];
                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));
                // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
                if(deltaR < -1 || deltaR > 1) {
                    continue;
                }
                // 根据亚像素精度偏移量delta调整最佳匹配索引
                //? 为什么用左图的图像尺度？　因为在视差计算时，是根据左图的uL（如下disparity = (uL-bestuR)）得来的
                float bestuR = mvScaleFactors[kpR.octave] * ((float)scaleduR0 + (float)bestincR + deltaR) ;
                // disparity:求得的视差值
                bestuR = bestuR * invFscale + Lcx - invFscale * Rcx ;
                float disparity = uL - bestuR ;
                if(disparity >= 0 && disparity < maxD) {
                    // 无穷远
                    if(disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    //if(iL % 16 == 0)
                    cout << "uL= " << uL << "\tuR= " << kpR.pt.x << "\tbestuR= " << bestuR << "\tdisparity = " << disparity << "\tDepth = " << mvDepth[iL] << endl;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                    mvuLeft[iL] = iL;
                    matchedLeftnums++;
                    LeftIdtoRightId[iL] = bestIdxR; // 左特征点对应右特征点ID,IL位置维IR
                    RightIdToLeftId[bestIdxR] = bestIdxR; // 右特征点ID
                }
            }
        }//遍历左图每个特征点
        // Step 6. 删除离缺点(outliers)
        // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
        // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4 * median;
        for(int i = vDistIdx.size() - 1; i >= 0; i--) {
            if(vDistIdx[i].first < thDist) {
                break;
            } else {
                matchedLeftnums--;
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
                mvuLeft[vDistIdx[i].second] = -1;
                RightIdToLeftId[LeftIdtoRightId[vDistIdx[i].second]] = -1;
                LeftIdtoRightId[vDistIdx[i].second] = -1; //删除左图离缺点,标志为-1
            }
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);
        for(int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];
            const float &v = kp.pt.y;
            const float &u = kp.pt.x;
            const float d = imDepth.at<float>(v, u);
            if(d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if(z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            // 由相机坐标系转换到世界坐标系
            // Twc为相机坐标系到世界坐标系的变换矩阵，mRcw = mTcw.rowRange(0,3).colRange(0,3);
            // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
            return mRwc * x3Dc + mOw;
        } else {
            return cv::Mat();
        }
    }

    bool Frame::imuIsPreintegrated() {
        unique_lock<std::mutex> lock(*mpMutexImu);
        return mbImuPreintegrated;
    }

    void Frame::setIntegrated() {
        unique_lock<std::mutex> lock(*mpMutexImu);
        mbImuPreintegrated = true;
    }

    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, GeometricCamera *pCamera2, cv::Mat &Tlr, Frame *pPrevF, const IMU::Calib &ImuCalib)
        : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mTlr(Tlr) {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        imgLeft = imLeft.clone();
        imgRight = imRight.clone();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // Frame ID
        mnId = nNextId++;
        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
        // ORB extraction
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 511);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 511);
        threadLeft.join();
        threadRight.join();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        Nleft = mvKeys.size();
        Nright = mvKeysRight.size();
        N = Nleft + Nright;
        if(N == 0) {
            return;
        }
        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations) {
            ComputeImageBounds(imLeft);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);
            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;
            mbInitialComputations = false;
        }
        mb = mbf / fx;
        mRlr = mTlr.rowRange(0, 3).colRange(0, 3);
        mtlr = mTlr.col(3);
        cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat trl = Rrl * (-1 * mTlr.col(3));
        cv::hconcat(Rrl, trl, mTrl);
        ComputeStereoFishEyeMatches();
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        //Put all descriptors in the same matrix
        cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(nullptr));
        mvbOutlier = vector<bool>(N, false);
        AssignFeaturesToGrid();
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        mpMutexImu = new std::mutex();
        UndistortKeyPoints();
        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
        double t_read = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t1 - t0).count();
        double t_orbextract = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1).count();
        double t_stereomatches = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t3 - t2).count();
        double t_assign = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t4 - t3).count();
        double t_undistort = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t5 - t4).count();
    }

    void Frame::ComputeStereoFishEyeMatches() {
        //Speed it up by matching keypoints in the lapping area
        vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
        vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());
        cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
        cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);
        mvLeftToRightMatch = vector<int>(Nleft, -1);
        mvRightToLeftMatch = vector<int>(Nright, -1);
        mvDepth = vector<float>(Nleft, -1.0f);
        mvuRight = vector<float>(Nleft, -1.0f);
        mvStereo3Dpoints = vector<cv::Mat>(Nleft);
        mnCloseMPs = 0;
        //Perform a brute force between Keypoint in the left and right image
        vector<vector<cv::DMatch>> matches;
        BFmatcher.knnMatch(stereoDescLeft, stereoDescRight, matches, 2);
        int nMatches = 0;
        int descMatches = 0;
        //Check matches using Lowe's ratio
        for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it) {
            if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7) {
                //For every good match, check parallax and reprojection error to discard spurious matches
                cv::Mat p3D;
                descMatches++;
                float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
                float depth = static_cast<KannalaBrandt8 *>(mpCamera)->TriangulateMatches(mpCamera2, mvKeys[(*it)[0].queryIdx + monoLeft], mvKeysRight[(*it)[0].trainIdx + monoRight], mRlr, mtlr, sigma1, sigma2, p3D);
                if(depth > 0.0001f) {
                    mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                    mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                    mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D.clone();
                    mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                    nMatches++;
                }
            }
        }
    }

    bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();
        cv::Mat mR, mt, twc;
        if(bRight) {
            cv::Mat Rrl = mTrl.colRange(0, 3).rowRange(0, 3);
            cv::Mat trl = mTrl.col(3);
            mR = Rrl * mRcw;
            mt = Rrl * mtcw + trl;
            twc = mRwc * mTlr.rowRange(0, 3).col(3) + mOw;
        } else {
            mR = mRcw;
            mt = mtcw;
            twc = mOw;
        }
        // 3D in camera coordinates
        cv::Mat Pc = mR * P + mt;
        const float Pc_dist = cv::norm(Pc);
        const float &PcZ = Pc.at<float>(2);
        // Check positive depth
        if(PcZ < 0.0f) {
            return false;
        }
        // Project in image and check it is not outside
        cv::Point2f uv;
        if(bRight) {
            uv = mpCamera2->project(Pc);
        } else {
            uv = mpCamera->project(Pc);
        }
        if(uv.x < mnMinX || uv.x > mnMaxX) {
            return false;
        }
        if(uv.y < mnMinY || uv.y > mnMaxY) {
            return false;
        }
        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - twc;
        const float dist = cv::norm(PO);
        if(dist < minDistance || dist > maxDistance) {
            return false;
        }
        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();
        const float viewCos = PO.dot(Pn) / dist;
        if(viewCos < viewingCosLimit) {
            return false;
        }
        
        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);
        if(bRight) {
            pMP->mTrackProjXR = uv.x;
            pMP->mTrackProjYR = uv.y;
            pMP->mnTrackScaleLevelR = nPredictedLevel;
            pMP->mTrackViewCosR = viewCos;
            pMP->mTrackDepthR = Pc_dist;
        } else {
            pMP->mTrackProjX = uv.x;
            pMP->mTrackProjY = uv.y;
            pMP->mnTrackScaleLevel = nPredictedLevel;
            pMP->mTrackViewCos = viewCos;
            pMP->mTrackDepth = Pc_dist;
        }
        return true;
    }

    cv::Mat Frame::UnprojectStereoFishEye(const int &i) {
        return mRwc * mvStereo3Dpoints[i] + mOw;
    }

    float Frame::max1(float &a, float &b) {
        return a > b ? a : b;
    }
} //namespace ORB_SLAM
