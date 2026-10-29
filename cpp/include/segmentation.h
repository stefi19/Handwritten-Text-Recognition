#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

Mat removeSmallComponents(const Mat& binary, int minArea = 3);
vector<int> computeHorizontalProjection(const Mat& binary);
vector<Rect> segmentLines(const Mat& binary, int minRowPixels = 3,
                           int maxGap = 5, int minLineHeight = 8,
                           int minLinePixels = 40, int componentGap = 8,
                           int margin = 3, double rowThresholdRatio = 0.05);
Mat normalizeLine(const Mat& gray, const Mat& binary, Rect region,
                   int targetHeight = 32, int padding = 2);
Mat drawHorizontalProjection(const vector<int>& projection, int width = 400);
Mat drawDetectedLines(const Mat& gray, const vector<Rect>& lines);
