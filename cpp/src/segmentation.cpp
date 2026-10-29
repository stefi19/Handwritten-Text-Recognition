#include "segmentation.h"

Mat removeSmallComponents(const Mat& binary, int minArea) {
    Mat labels, stats, centroids;
    int count = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    vector<bool> keep(count, false);
    for (int i = 1; i < count; i++) {
        keep[i] = stats.at<int>(i, CC_STAT_AREA) >= minArea;
    }

    Mat clean = Mat::zeros(binary.size(), CV_8UC1);
    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            int label = labels.at<int>(y, x);
            if (keep[label]) {
                clean.at<uchar>(y, x) = 255;
            }
        }
    }
    return clean;
}

vector<int> computeHorizontalProjection(const Mat& binary) {
    vector<int> projection(binary.rows, 0);
    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            if (binary.at<uchar>(y, x) != 0) {
                projection[y]++;
            }
        }
    }
    return projection;
}

vector<Rect> segmentLines(const Mat& binary, int minRowPixels, int maxGap,
                           int minLineHeight, int minLinePixels, int componentGap,
                           int margin, double rowThresholdRatio) {
    vector<int> projection = computeHorizontalProjection(binary);
    int peak = 0;
    for (int row = 0; row < binary.rows; row++) {
        peak = max(peak, projection[row]);
    }
    // Sparse ascenders and descenders can fill the valley between line bodies.
    int rowThreshold = max(minRowPixels, cvRound(peak * rowThresholdRatio));
    vector<Rect> bands;
    int y = 0;
    while (y < binary.rows) {
        if (projection[y] < rowThreshold) {
            y++;
            continue;
        }

        int start = y;
        int last = y;
        y++;
        while (y < binary.rows) {
            if (projection[y] >= rowThreshold) {
                last = y;
            } else if (y - last > maxGap) {
                break;
            }
            y++;
        }

        int pixels = 0;
        for (int row = start; row <= last; row++) {
            pixels += projection[row];
        }
        int height = last - start + 1;
        if (height >= minLineHeight && pixels >= minLinePixels) {
            bands.push_back(Rect(0, start, binary.cols, height));
        }
    }

    Mat labels, stats, centroids;
    int count = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // Keep a component whole if it bridges two projection bands. Touching lines
    // can remain merged: a horizontal cut here could remove parts of letters.
    for (int i = 1; i < count; i++) {
        int top = stats.at<int>(i, CC_STAT_TOP);
        int bottom = top + stats.at<int>(i, CC_STAT_HEIGHT);
        int first = -1;
        int last = -1;
        for (int j = 0; j < (int)bands.size(); j++) {
            if (top < bands[j].y + bands[j].height && bottom > bands[j].y) {
                if (first == -1) {
                    first = j;
                }
                last = j;
            }
        }
        if (first != -1 && last > first) {
            int end = bands[last].y + bands[last].height;
            bands[first].height = end - bands[first].y;
            bands.erase(bands.begin() + first + 1, bands.begin() + last + 1);
        }
    }

    vector<Rect> bounds(bands.size());
    vector<Rect> detached;
    for (int i = 1; i < count; i++) {
        Rect component(stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP),
                       stats.at<int>(i, CC_STAT_WIDTH), stats.at<int>(i, CC_STAT_HEIGHT));
        int best = -1;
        int bestOverlap = 0;
        for (int j = 0; j < (int)bands.size(); j++) {
            int overlap = min(component.y + component.height, bands[j].y + bands[j].height)
                          - max(component.y, bands[j].y);
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                best = j;
            }
        }
        if (best == -1) {
            detached.push_back(component);
        } else {
            if (bounds[best].empty()) {
                bounds[best] = component;
            } else {
                bounds[best] = bounds[best] | component;
            }
        }
    }

    // Attach dots and punctuation after full letter heights have been restored.
    for (int i = 0; i < (int)detached.size(); i++) {
        Rect component = detached[i];
        int best = -1;
        int bestGap = componentGap + 1;
        for (int j = 0; j < (int)bounds.size(); j++) {
            if (bounds[j].empty()) {
                continue;
            }
            int gap = max(0, max(bounds[j].y - component.y - component.height,
                                 component.y - bounds[j].y - bounds[j].height));
            if (gap < bestGap) {
                bestGap = gap;
                best = j;
            }
        }
        if (best != -1) {
            bounds[best] = bounds[best] | component;
        }
    }

    vector<Rect> lines;
    for (int i = 0; i < (int)bounds.size(); i++) {
        if (bounds[i].empty()) {
            continue;
        }
        lines.push_back(bounds[i]);
    }

    // Divide whitespace between neighbors so margins do not include another line.
    vector<Rect> padded;
    for (int i = 0; i < (int)lines.size(); i++) {
        int topLimit = 0;
        int bottomLimit = binary.rows;
        if (i > 0) {
            int middle = (lines[i - 1].y + lines[i - 1].height + lines[i].y) / 2;
            topLimit = min(lines[i].y, middle);
        }
        if (i + 1 < (int)lines.size()) {
            int middle = (lines[i].y + lines[i].height + lines[i + 1].y) / 2;
            bottomLimit = max(lines[i].y + lines[i].height, middle);
        }
        int left = max(0, lines[i].x - margin);
        int right = min(binary.cols, lines[i].x + lines[i].width + margin);
        int top = max(topLimit, lines[i].y - margin);
        int bottom = min(bottomLimit, lines[i].y + lines[i].height + margin);
        padded.push_back(Rect(left, top, right - left, bottom - top));
    }
    return padded;
}

Mat normalizeLine(const Mat& gray, const Mat& binary, Rect region,
                   int targetHeight, int padding) {
    CV_Assert(targetHeight > 2 * padding && padding >= 0);
    region = region & Rect(0, 0, gray.cols, gray.rows);
    int left = region.x + region.width;
    int right = -1;
    int top = region.y + region.height;
    int bottom = -1;
    for (int y = region.y; y < region.y + region.height; y++) {
        for (int x = region.x; x < region.x + region.width; x++) {
            if (binary.at<uchar>(y, x) != 0) {
                left = min(left, x);
                right = max(right, x);
                top = min(top, y);
                bottom = max(bottom, y);
            }
        }
    }
    if (right == -1) {
        return Mat();
    }

    // Keep up to two source pixels around the mask for antialiased stroke edges.
    left = max(region.x, left - 2);
    right = min(region.x + region.width - 1, right + 2);
    top = max(region.y, top - 2);
    bottom = min(region.y + region.height - 1, bottom + 2);
    Rect crop(left, top, right - left + 1, bottom - top + 1);
    int height = targetHeight - 2 * padding;
    double scale = (double)height / crop.height;
    int width = max(1, cvRound(crop.width * scale));
    Mat resized;
    int interpolation = scale < 1 ? INTER_AREA : INTER_CUBIC;
    resize(gray(crop), resized, Size(width, height), 0, 0, interpolation);
    Mat normalized;
    copyMakeBorder(resized, normalized, padding, padding, padding, padding,
                   BORDER_CONSTANT, Scalar(255));
    return normalized;
}

Mat drawHorizontalProjection(const vector<int>& projection, int width) {
    Mat histogram((int)projection.size(), width, CV_8UC3, Scalar(255, 255, 255));
    int maxPixels = 0;
    for (int y = 0; y < (int)projection.size(); y++) {
        maxPixels = max(maxPixels, projection[y]);
    }
    if (maxPixels == 0) {
        return histogram;
    }
    for (int y = 0; y < (int)projection.size(); y++) {
        int length = cvRound((double)projection[y] * (width - 1) / maxPixels);
        if (projection[y] > 0) {
            line(histogram, Point(0, y), Point(length, y), Scalar(60, 60, 60));
        }
    }
    return histogram;
}

Mat drawDetectedLines(const Mat& gray, const vector<Rect>& lines) {
    Mat display;
    cvtColor(gray, display, COLOR_GRAY2BGR);
    for (int i = 0; i < (int)lines.size(); i++) {
        rectangle(display, lines[i], Scalar(0, 0, 255), 1);
        Point position(lines[i].x, max(12, lines[i].y - 3));
        putText(display, to_string(i), position, FONT_HERSHEY_SIMPLEX,
                0.4, Scalar(0, 0, 255), 1);
    }
    return display;
}
