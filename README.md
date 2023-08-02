# Preliminary
This is a developing project for paper --"A Multi-Focal Dynamic SLAM"
It is built upon the ORB-SLAM3 project.
The code in the project is not concise enough, so it may be hard to read.

# MF-DynaSLAM
**Authors:** Mingchi Feng, Xuan Yi, Kun Wang

In order to achieve wider perceptual field of view, longer measuring distance and higher positioning accuracy, our multi-focal stereo dynamic visual simultaneous localization and mapping (SLAM) technology uses an effective feature extraction and matching method based on adaptive image pyramid, and deep learning segmentation method is adopted to obtain priori dynamic objects, while multi-view geometry, regional feature flow, and inverse perspective mapping are combined to verify dynamic objects and eliminate dynamic features. Finally compared with ORB-SLAM3 and DynaSLAM, it has higher accuracy, larger perceived field of view, longer measuring distance and real-time performance.


#  Prerequisites
We have tested the library in  **20.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 3.0. Tested with OpenCV 3.4.5**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

#  Building and examples

```
We provide a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM3*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd MF-DynaSLAM
chmod +x build.sh
./build.sh
```

This will create **libMF-DynaSLAM.so**  at *lib* folder and the executables in *Examples/stereo_test*  in Examples folder.




