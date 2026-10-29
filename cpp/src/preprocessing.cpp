#include "preprocessing.h"

#include <cmath>
#include <vector>

Mat convertToGrayscale(const Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    return gray;
}

Mat removeNoise(const Mat& gray) {
    Mat denoised;
    GaussianBlur(gray, denoised, Size(3, 3), 0);
    return denoised;
}

Mat binarizeImage(const Mat& gray, bool useOtsu, int blockSize, double thresholdC) {
    Mat binary;
    if (useOtsu) {
        threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    } else {
        adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY_INV, blockSize, thresholdC);
    }
    return binary;
}

static double computeProjectionScore(const vector<Point>& points, Size size,
                                     double angle) {
    double radians = angle * CV_PI / 180.0;
    double sine = sin(radians);
    double cosine = cos(radians);
    double centerX = (size.width - 1) / 2.0;
    double centerY = (size.height - 1) / 2.0;
    int height = (int)ceil(hypot(size.width, size.height)) + 2;
    vector<int> projection(height, 0);

    for (int i = 0; i < (int)points.size(); i++) {
        double x = points[i].x - centerX;
        double y = points[i].y - centerY;
        int row = cvRound(-sine * x + cosine * y + (height - 1) / 2.0);
        projection[row]++;
    }

    // Aligned text concentrates foreground into fewer rows, increasing sum(H[y]^2).
    double score = 0;
    for (int y = 0; y < height; y++) {
        score += (double)projection[y] * projection[y];
    }
    return score;
}

double estimateSkewAngle(const Mat& binary, double maxAngle, double angleStep,
                         int maxWidth) {
    CV_Assert(maxAngle >= 0 && maxAngle <= 45 && angleStep > 0 && maxWidth > 0);
    Mat small = binary;
    if (binary.cols > maxWidth) {
        double scale = (double)maxWidth / binary.cols;
        int height = max(1, cvRound(binary.rows * scale));
        resize(binary, small, Size(maxWidth, height), 0, 0, INTER_NEAREST);
    }

    vector<Point> points;
    for (int y = 0; y < small.rows; y++) {
        for (int x = 0; x < small.cols; x++) {
            if (small.at<uchar>(y, x) != 0) {
                points.push_back(Point(x, y));
            }
        }
    }
    if (points.size() < 40 || maxAngle == 0) {
        return 0;
    }

    double bestAngle = 0;
    double bestScore = computeProjectionScore(points, small.size(), 0);
    for (double angle = -maxAngle; angle <= maxAngle + 0.0001; angle += angleStep) {
        double score = computeProjectionScore(points, small.size(), angle);
        if (score > bestScore) {
            bestScore = score;
            bestAngle = angle;
        }
    }

    double start = max(-maxAngle, bestAngle - angleStep);
    double end = min(maxAngle, bestAngle + angleStep);
    for (double angle = start; angle <= end + 0.0001; angle += angleStep / 5) {
        double score = computeProjectionScore(points, small.size(), angle);
        if (score > bestScore) {
            bestScore = score;
            bestAngle = angle;
        }
    }
    return bestAngle;
}

void deskewImage(const Mat& gray, const Mat& binary, double angle,
                 Mat& deskewedGray, Mat& deskewedBinary) {
    if (abs(angle) < 0.01) {
        deskewedGray = gray.clone();
        deskewedBinary = binary.clone();
        return;
    }

    Point2f center((gray.cols - 1) / 2.0f, (gray.rows - 1) / 2.0f);
    Mat rotation = getRotationMatrix2D(center, angle, 1);
    double cosine = abs(rotation.at<double>(0, 0));
    double sine = abs(rotation.at<double>(0, 1));
    int width = (int)ceil(gray.cols * cosine + gray.rows * sine) + 2;
    int height = (int)ceil(gray.rows * cosine + gray.cols * sine) + 2;
    rotation.at<double>(0, 2) += (width - 1) / 2.0 - center.x;
    rotation.at<double>(1, 2) += (height - 1) / 2.0 - center.y;

    warpAffine(gray, deskewedGray, rotation, Size(width, height),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
    warpAffine(binary, deskewedBinary, rotation, Size(width, height),
               INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
}

Mat closeSmallGaps(const Mat& binary) {
    Mat closed;
    // A horizontal kernel repairs small stroke gaps without vertical dilation.
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);
    return closed;
}
