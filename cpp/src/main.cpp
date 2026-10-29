#include "preprocessing.h"
#include "segmentation.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

void printUsage() {
    cout << "Usage: ./htr_preprocess --input image.png --output_dir output\n"
         << "       [--threshold adaptive|otsu] [--no-deskew] [--close]\n";
}

void saveImage(const fs::path& path, const Mat& image) {
    if (!imwrite(path.string(), image)) {
        throw runtime_error("Could not save image: " + path.string());
    }
}

int main(int argc, char** argv) {
    string inputPath;
    string outputDir = "output";
    bool useOtsu = false;
    bool useDeskew = true;
    bool useClosing = false;

    for (int i = 1; i < argc; i++) {
        string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        }
        if (argument == "--no-deskew") {
            useDeskew = false;
        } else if (argument == "--close") {
            useClosing = true;
        } else if ((argument == "--input" || argument == "--output_dir" ||
                    argument == "--threshold") && i + 1 < argc) {
            string value = argv[++i];
            if (argument == "--input") {
                inputPath = value;
            } else if (argument == "--output_dir") {
                outputDir = value;
            } else if (value == "otsu") {
                useOtsu = true;
            } else if (value == "adaptive") {
                useOtsu = false;
            } else {
                cerr << "Threshold must be adaptive or otsu.\n";
                return 1;
            }
        } else {
            cerr << "Unknown option or missing value: " << argument << '\n';
            printUsage();
            return 1;
        }
    }
    if (inputPath.empty() || outputDir.empty()) {
        printUsage();
        return 1;
    }

    const int blockSize = 31;
    const double thresholdC = 15;
    const double maxSkewAngle = 12;
    const double angleStep = 0.5;
    const int minComponentArea = 3;
    const int minRowPixels = 3;
    const double rowThresholdRatio = 0.05;
    const int maxGap = 5;
    const int minLineHeight = 8;
    const int minLinePixels = 40;
    const int componentGap = 8;
    const int cropMargin = 3;
    const int targetHeight = 32;
    const int padding = 2;

    try {
        Mat image = imread(inputPath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Could not read image: " << inputPath << '\n';
            return 1;
        }
        Mat gray = convertToGrayscale(image);
        Mat denoised = removeNoise(gray);
        Mat binary = binarizeImage(denoised, useOtsu, blockSize, thresholdC);
        Mat clean = removeSmallComponents(binary, minComponentArea);
        double angle = 0;
        if (useDeskew) {
            angle = estimateSkewAngle(clean, maxSkewAngle, angleStep);
        }

        Mat deskewedGray, deskewedBinary;
        deskewImage(denoised, clean, angle, deskewedGray, deskewedBinary);
        if (useClosing) {
            deskewedBinary = closeSmallGaps(deskewedBinary);
        }
        deskewedBinary = removeSmallComponents(deskewedBinary, minComponentArea);
        vector<int> projection = computeHorizontalProjection(deskewedBinary);
        vector<Rect> lines = segmentLines(deskewedBinary, minRowPixels, maxGap,
                                          minLineHeight, minLinePixels, componentGap,
                                          cropMargin, rowThresholdRatio);

        fs::path debugDir = fs::path(outputDir) / "debug";
        fs::path linesDir = fs::path(outputDir) / "lines";
        fs::create_directories(debugDir);
        fs::create_directories(linesDir);

        // Remove only previous numbered line exports, including after a blank input.
        for (const fs::directory_entry& entry : fs::directory_iterator(linesDir)) {
            string name = entry.path().filename().string();
            bool numberedLine = name.size() > 9 && name.substr(0, 5) == "line_" &&
                                name.substr(name.size() - 4) == ".png";
            for (int i = 5; numberedLine && i < (int)name.size() - 4; i++) {
                if (name[i] < '0' || name[i] > '9') {
                    numberedLine = false;
                }
            }
            if (entry.is_regular_file() && numberedLine) {
                fs::remove(entry.path());
            }
        }

        saveImage(debugDir / "01_grayscale.png", gray);
        saveImage(debugDir / "02_denoised.png", denoised);
        saveImage(debugDir / "03_binary.png", binary);
        saveImage(debugDir / "04_deskewed.png", deskewedGray);
        saveImage(debugDir / "04_deskewed_binary.png", deskewedBinary);
        saveImage(debugDir / "05_projection.png", drawHorizontalProjection(projection));
        saveImage(debugDir / "06_detected_lines.png", drawDetectedLines(deskewedGray, lines));

        for (int i = 0; i < (int)lines.size(); i++) {
            Mat normalized = normalizeLine(deskewedGray, deskewedBinary, lines[i],
                                           targetHeight, padding);
            ostringstream name;
            name << "line_" << setw(2) << setfill('0') << i << ".png";
            saveImage(linesDir / name.str(), normalized);
        }

        cout << "Deskew correction angle: " << fixed << setprecision(2) << angle << " degrees\n";
        cout << "Total detected lines: " << lines.size() << '\n';
        cout << "Saved results to: " << outputDir << '\n';
    } catch (const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
