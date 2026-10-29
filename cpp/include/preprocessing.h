#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat convertToGrayscale(const Mat& image);
Mat removeNoise(const Mat& gray);
Mat binarizeImage(const Mat& gray, bool useOtsu = false,
                  int blockSize = 31, double thresholdC = 15);
double estimateSkewAngle(const Mat& binary, double maxAngle = 12,
                         double angleStep = 0.5, int maxWidth = 1000);
void deskewImage(const Mat& gray, const Mat& binary, double angle,
                 Mat& deskewedGray, Mat& deskewedBinary);
Mat closeSmallGaps(const Mat& binary);
