#ifndef NCC
#define NCC

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include "Frame.h"
using namespace cv;

namespace ORB_SLAM3 {

    // static Point* upleft ,* boright;

    void ncclong(Mat &srcImage, Mat &templImage, Mat &result);

    /**
     * @brief FindPoints NCC looking for corners
     * @param srcImage   Short focal length image
     * @param templImage Long focal length image
     * @param scale Left_fx / Right_fx;
     */
    void FindPoints(Mat &srcImage, Mat &templImage, float scale);

}

#endif
