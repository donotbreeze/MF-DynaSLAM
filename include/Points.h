#ifndef POINTS_H
#define POINTS_H
#include<iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<string>
using namespace cv;
using namespace std;
namespace ORB_SLAM3 {

    extern float matchdistance, perspectdistance, objbestdistance;
    extern cv::Mat LEFTD, LEFTK, LEFTR, LEFTP;
    extern cv::Mat RIGHTD, RIGHTK, RIGHTR, RIGHTP;

    // 计算H矩阵
    extern cv::Mat H, Hlast;
    // 前后帧之间的基础矩阵Ｆ
    extern cv::Mat F;
    // 前后关键帧之间的基础矩阵Ｆ
    extern cv::Mat Fkeyframe;

    extern int squenceimg;//图像序列
    extern Point LeftUp, RightBottom;//roi角点
    extern float Fscale;//焦距
    extern float invFscale;
    extern int height;
    extern int width;
    extern float fx;
    extern float Rfx;
    extern float Lcx;
    extern float Lcy;
    extern float Rcx;
    extern float Rcy;
    extern float CamY;
    // 层数
    extern float MFnLevels;

    extern double  duration;
    extern double  averduration;
    extern double  allduration;
    extern int summatchimg;
    extern int matchednums; //匹配的数
    extern float myscaleFactor;
    void Calibfind(const string &pathset);
    void Nccfind(const string &pathset, vector<string> &vstrImageLeft, vector<string> &vstrImageRight);

    void readcameraMat(const string &pathset);
}
#endif


