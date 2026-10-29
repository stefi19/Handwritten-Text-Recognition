#include "preprocessing.h"
#include "segmentation.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

void check(bool condition, const string& message) {
    if (!condition) {
        throw runtime_error(message);
    }
}

void testProjectionAndNoise() {
    Mat canvas = Mat::zeros(30, 40, CV_8UC1);
    Mat binary = canvas(Rect(4, 3, 20, 20));
    rectangle(binary, Rect(2, 5, 8, 3), Scalar(255), FILLED);
    binary.at<uchar>(0, 0) = 255;
    binary.at<uchar>(18, 18) = 255;
    binary.at<uchar>(18, 19) = 255;
    Mat clean = removeSmallComponents(binary, 3);
    check(countNonZero(clean) == 24, "Small component removal");
    check(countNonZero(binary) == 27, "Component removal must preserve input");
    vector<int> projection = computeHorizontalProjection(clean);
    for (int y = 0; y < clean.rows; y++) {
        int expected = y >= 5 && y < 8 ? 8 : 0;
        check(projection[y] == expected, "Explicit horizontal projection");
    }
    projection = computeHorizontalProjection(binary);
    check(projection[18] == 2, "Projection on a non-contiguous ROI");
}

void testThresholds() {
    Mat image(100, 200, CV_8UC3, Scalar(255, 255, 255));
    rectangle(image, Rect(30, 30, 80, 20), Scalar(0, 0, 0), FILLED);
    Mat gray = convertToGrayscale(image);
    Mat denoised = removeNoise(gray);
    for (int mode = 0; mode < 2; mode++) {
        Mat binary = binarizeImage(denoised, mode == 1);
        check(binary.at<uchar>(30, 50) == 255, "Text must be white in binary mask");
        check(binary.at<uchar>(0, 0) == 0, "Background must be black in binary mask");
        Mat white(gray.size(), CV_8UC1, Scalar(255));
        check(countNonZero(binarizeImage(white, mode == 1)) == 0, "Blank threshold");
    }
}

void testLineBounds() {
    Mat binary = Mat::zeros(140, 240, CV_8UC1);
    rectangle(binary, Rect(20, 22, 150, 18), Scalar(255), FILLED);
    rectangle(binary, Rect(35, 58, 65, 18), Scalar(255), FILLED);
    rectangle(binary, Rect(10, 99, 190, 18), Scalar(255), FILLED);
    rectangle(binary, Rect(24, 5, 3, 3), Scalar(255), FILLED);
    rectangle(binary, Rect(50, 10, 2, 13), Scalar(255), FILLED);
    rectangle(binary, Rect(40, 40, 2, 6), Scalar(255), FILLED);
    binary.at<uchar>(130, 230) = 255;
    Mat clean = removeSmallComponents(binary);
    vector<Rect> lines = segmentLines(clean);
    check(lines.size() == 3, "Three separate line candidates");
    check(lines[0].contains(Point(24, 5)), "Attach dots using restored component heights");
    check(lines[0].contains(Point(40, 45)), "Sparse descender must not be cut");
    for (int i = 1; i < (int)lines.size(); i++) {
        check(lines[i - 1].y + lines[i - 1].height <= lines[i].y,
              "Reading order and non-overlapping line margins");
    }

    Mat gray(binary.size(), CV_8UC1, Scalar(255));
    gray.setTo(Scalar(70), clean);
    Mat first = normalizeLine(gray, clean, lines[0]);
    Mat second = normalizeLine(gray, clean, lines[1]);
    check(first.rows == 32 && second.rows == 32 && first.type() == CV_8UC1,
          "Normalization height and grayscale format");
    check(first.cols > second.cols, "Normalization must preserve variable widths");
    check(first.at<uchar>(0, 0) == 255 && first.at<uchar>(31, 0) == 255,
          "Normalization white padding");
    double minimum, maximum;
    minMaxLoc(first, &minimum, &maximum);
    check(minimum > 0 && minimum < 100 && maximum == 255,
          "Normalization must use grayscale rather than binary strokes");
    check(normalizeLine(gray, clean, Rect(210, 0, 20, 10)).empty(), "Empty crop");

    Mat wideGray(50, 100, CV_8UC1, Scalar(70));
    Mat wideMask(50, 100, CV_8UC1, Scalar(255));
    Mat wideLine = normalizeLine(wideGray, wideMask, Rect(0, 0, 100, 50));
    check(wideLine.cols == 60, "A 2:1 crop must remain 2:1 inside the padding");
}

void testGapsAndBorders() {
    Mat binary = Mat::zeros(70, 100, CV_8UC1);
    rectangle(binary, Rect(0, 0, 50, 10), Scalar(255), FILLED);
    rectangle(binary, Rect(0, 14, 50, 10), Scalar(255), FILLED);
    rectangle(binary, Rect(30, 50, 70, 20), Scalar(255), FILLED);
    vector<Rect> lines = segmentLines(binary);
    check(lines.size() == 2, "Four-row gap inside a line");
    check(lines[0].y == 0 && lines[0].x == 0, "Top and left document borders");
    check(lines[1].y + lines[1].height == binary.rows &&
          lines[1].x + lines[1].width == binary.cols, "Bottom and right document borders");

    Mat closeLines = Mat::zeros(60, 100, CV_8UC1);
    rectangle(closeLines, Rect(10, 5, 70, 15), Scalar(255), FILLED);
    rectangle(closeLines, Rect(10, 26, 70, 15), Scalar(255), FILLED);
    lines = segmentLines(closeLines);
    check(lines.size() == 2, "Six-row inter-line gap must remain separate");
    check(lines[0].y + lines[0].height <= lines[1].y, "Clipped adjacent margins");

    Mat bridge = Mat::zeros(80, 100, CV_8UC1);
    rectangle(bridge, Rect(10, 10, 60, 15), Scalar(255), FILLED);
    rectangle(bridge, Rect(10, 50, 60, 15), Scalar(255), FILLED);
    line(bridge, Point(20, 24), Point(20, 50), Scalar(255));
    check(segmentLines(bridge).size() == 1, "Keep a component spanning two bands whole");
    check(segmentLines(Mat::zeros(30, 30, CV_8UC1)).empty(), "Blank segmentation");

    Mat overlappingHeights = Mat::zeros(90, 240, CV_8UC1);
    rectangle(overlappingHeights, Rect(10, 10, 200, 15), Scalar(255), FILLED);
    rectangle(overlappingHeights, Rect(10, 25, 2, 16), Scalar(255), FILLED);
    rectangle(overlappingHeights, Rect(20, 48, 200, 15), Scalar(255), FILLED);
    rectangle(overlappingHeights, Rect(215, 35, 2, 14), Scalar(255), FILLED);
    lines = segmentLines(overlappingHeights);
    check(lines.size() == 2, "Low-density valley between disconnected lines");
    check(lines[0].contains(Point(10, 40)) && lines[1].contains(Point(215, 35)),
          "Overlapping component heights must preserve full strokes");
}

void testRotationAndSkew() {
    Mat gray(280, 800, CV_8UC1, Scalar(255));
    putText(gray, "Handwritten text recognition", Point(35, 60),
            FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, Scalar(40), 2, LINE_AA);
    putText(gray, "A short second line", Point(35, 140),
            FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, Scalar(40), 2, LINE_AA);
    putText(gray, "Letters with ascenders and descenders", Point(35, 220),
            FONT_HERSHEY_SCRIPT_SIMPLEX, 1.1, Scalar(40), 2, LINE_AA);
    Mat binary = binarizeImage(removeNoise(gray), true);
    for (int i = 0; i < 2; i++) {
        double angle = i == 0 ? -7 : 7;
        Mat rotatedGray, rotatedBinary;
        deskewImage(gray, binary, angle, rotatedGray, rotatedBinary);
        double correction = estimateSkewAngle(rotatedBinary);
        cout << "Synthetic rotation " << angle << ", estimated correction " << correction << '\n';
        check(abs(correction + angle) < 0.8, "Skew sign and angle estimate");
        Mat correctedGray, correctedBinary;
        deskewImage(rotatedGray, rotatedBinary, correction, correctedGray, correctedBinary);
        check(segmentLines(removeSmallComponents(correctedBinary)).size() == 3,
              "Three lines after skew correction");
        if (i == 0) {
            std::filesystem::create_directories("validation");
            imwrite("validation/synthetic.png", rotatedGray);
            imwrite("validation/blank.png", Mat(100, 200, CV_8UC1, Scalar(255)));
        }
    }

    Mat corners = Mat::zeros(100, 200, CV_8UC1);
    rectangle(corners, Rect(0, 0, 10, 10), Scalar(255), FILLED);
    rectangle(corners, Rect(190, 0, 10, 10), Scalar(255), FILLED);
    rectangle(corners, Rect(0, 90, 10, 10), Scalar(255), FILLED);
    rectangle(corners, Rect(190, 90, 10, 10), Scalar(255), FILLED);
    Mat cornerGray = 255 - corners;
    Mat rotatedGray, rotatedBinary;
    deskewImage(cornerGray, corners, 12, rotatedGray, rotatedBinary);
    check(rotatedBinary.cols > corners.cols && rotatedBinary.rows > corners.rows,
          "Rotation must expand the canvas");
    check(abs(countNonZero(rotatedBinary) - 400) <= 8, "Rotation must retain corner content");
    for (int y = 0; y < rotatedBinary.rows; y++) {
        for (int x = 0; x < rotatedBinary.cols; x++) {
            if (rotatedBinary.at<uchar>(y, x) != 0) {
                check(rotatedGray.at<uchar>(y, x) < 220, "Gray and mask rotation alignment");
            }
        }
    }
    check(estimateSkewAngle(Mat::zeros(20, 20, CV_8UC1)) == 0, "Blank skew estimate");
}

int main() {
    try {
        testProjectionAndNoise();
        testThresholds();
        testLineBounds();
        testGapsAndBorders();
        testRotationAndSkew();
        cout << "All pipeline checks passed.\n";
    } catch (const exception& error) {
        cerr << "Failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
