# Handwritten Text Recognition — C++ preprocessing

OpenCV 4 preprocessing and horizontal-projection line segmentation for dark English handwriting on light paper. The output is ordered grayscale line images, each 32 pixels high with a variable width, for a separately trained recognition model.

The standalone functions, explicit pixel loops, and OpenCV types follow the university lab code in [Peek-a-Cube](https://github.com/stefi19/Peek-a-Cube/blob/main/main.cpp). Its basic CMake/OpenCV setup is reused, with an explicit source list and a portable OpenCV search. The local repository was empty, so there was no existing implementation to replace. C++17 is used for creating output directories with `std::filesystem`.

## Build and run on macOS

From this repository directory:

```bash
brew install cmake opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
cmake --build build -j
./build/htr_preprocess --input samples/document.png --output_dir output
```

Supply your own image at `samples/document.png`, or change the input path. Paths containing spaces should be quoted. If CMake already finds OpenCV, `-DOpenCV_DIR=...` can be omitted.

Alternative threshold and processing options:

```bash
./build/htr_preprocess --input samples/document.png --output_dir output --threshold otsu
./build/htr_preprocess --input samples/document.png --output_dir output --no-deskew
./build/htr_preprocess --input samples/document.png --output_dir output --close
./build/htr_preprocess --help
```

Adaptive thresholding and deskewing are enabled by default. Closing is disabled by default. The program prints the applied correction angle and detected line count. Missing/unreadable inputs and failed writes return a nonzero exit code. A blank page produces zero lines.

## Files

```text
cpp/
├── include/
│   ├── preprocessing.h
│   └── segmentation.h
└── src/
    ├── preprocessing.cpp
    ├── segmentation.cpp
    └── main.cpp
tests/
└── pipeline_tests.cpp
CMakeLists.txt
README.md
```

There are no application classes, recognition models, or Python scripts.

## Functions

| Function | Purpose |
| --- | --- |
| `convertToGrayscale()` | Converts the BGR image read by `imread()` into one grayscale channel with `cvtColor()`. |
| `removeNoise()` | Applies a 3×3 Gaussian blur. |
| `binarizeImage()` | Applies Gaussian adaptive thresholding or Otsu. Both use `THRESH_BINARY_INV`: text is 255 and background is 0. |
| `computeProjectionScore()` | Internal function that projects foreground points at a trial rotation and sums squared row counts. |
| `estimateSkewAngle()` | Searches correction angles from −12° to +12°, then searches more finely near the best angle. A smaller copy limits computation; the full-resolution image is retained. |
| `deskewImage()` | Rotates denoised grayscale and binary images with the same transform. Expands the canvas to preserve the document. Uses linear interpolation/white borders for grayscale and nearest-neighbor interpolation/black borders for the mask. |
| `closeSmallGaps()` | Optional 3×1 horizontal closing on the binary mask to repair tiny stroke gaps. It does not change the grayscale image. |
| `removeSmallComponents()` | Uses `connectedComponentsWithStats()` and explicit label-image loops to discard components below a small area threshold. Label 0 is always background. |
| `computeHorizontalProjection()` | Counts white foreground pixels in each row using nested loops. |
| `segmentLines()` | Finds projection bands, bridges short gaps, rejects weak candidates, and refines crops using component bounding boxes. |
| `normalizeLine()` | Finds mask bounds inside a line rectangle, retains nearby grayscale stroke edges, resizes with one scale factor, and adds white padding. |
| `drawHorizontalProjection()` | Draws horizontal bars at their corresponding document rows, scaled to the largest row count. |
| `drawDetectedLines()` | Draws numbered rectangles on the deskewed grayscale document. |
| `printUsage()`, `saveImage()`, `main()` | Handle command-line usage, checked image writes, and the processing sequence. |

## Segmentation algorithm

1. Remove connected components smaller than `minComponentArea`. Estimate skew from this cleaned mask, rotate the grayscale/mask pair, optionally close horizontal gaps, and remove tiny components again.
2. Explicitly count white pixels per row: `projection[y] = number of foreground pixels on row y`.
3. Treat a row as active when its count reaches `max(minRowPixels, round(peak * rowThresholdRatio))`. The small peak-relative threshold lets sparse ascenders and descenders form a low-density valley between line bodies.
4. Scan from top to bottom. Start a band at an active row and continue across up to `maxGap` consecutive weak rows. Reject bands with too little height or too few foreground pixels.
5. Compute connected-component statistics. If one component intersects multiple candidate bands, merge those bands to preserve the component. Otherwise, assign components to the band with the greatest vertical overlap. Restore full letter bounds first, then attach nearby detached components, such as dots and punctuation, to the nearest restored line within `componentGap` pixels.
6. Use the union of the assigned component boxes for each crop. This restores sparse ascenders and descenders excluded by the row threshold. Add a small margin, limiting it in the whitespace between neighboring lines. Component bounds are preserved even if neighboring letter heights overlap slightly.
7. Export in the same top-to-bottom order as the bands.

This remains a single-column projection method. A component physically connecting two lines can produce a merged crop. Disconnected lines whose projection valleys are too dense can also merge. Slightly overlapping component heights can cause rectangular crops to contain small fragments of a neighboring line. Connected components cannot reliably resolve these cases. Tables, multiple columns, ruled paper, perspective distortion, severe shadows, and curved baselines need additional processing or manual inspection.

## Normalization

The binary mask locates the handwriting; the actual saved pixels come from the deskewed, denoised grayscale image. Text remains dark on a light background.

`normalizeLine()` includes up to two source pixels around the detected mask bounds for antialiased stroke edges. With the defaults, it resizes this crop to a content height of 28 pixels, computes `width = round(cropWidth * 28 / cropHeight)`, and adds two white pixels on each side. The final height is exactly 32; the final width varies. Integer rounding is the only aspect-ratio approximation. Downscaling uses `INTER_AREA`; upscaling uses `INTER_CUBIC`.

The recognition model should be trained with the same height, polarity, padding, and normalization convention. These images have not been evaluated with a CRNN in this project.

## Debug output

```text
output/
├── debug/
│   ├── 01_grayscale.png
│   ├── 02_denoised.png
│   ├── 03_binary.png
│   ├── 04_deskewed.png
│   ├── 04_deskewed_binary.png
│   ├── 05_projection.png
│   └── 06_detected_lines.png
└── lines/
    ├── line_00.png
    ├── line_01.png
    └── ...
```

`03_binary.png` shows the threshold result before component removal. `04_deskewed.png` is the grayscale document used for crops; the extra `04_deskewed_binary.png` shows the exact mask used for segmentation. Histogram rows correspond to the deskewed document rows.

Reusing an output directory replaces the debug images and removes old numbered `line_<digits>.png` exports before saving the current lines. Other filenames are retained. Use a separate output directory for each document when preserving previous results.

## Parameters to tune

The application parameters are together in `main.cpp`. Defaults in the headers are also available when calling the functions independently. Pixel-based settings refer to the full-resolution image except for skew estimation's smaller working copy.

| Parameter | Default | Effect |
| --- | --- | --- |
| `blockSize` | 31 | Adaptive threshold neighborhood. Must be odd and greater than 1; adjust for stroke size and lighting variation. |
| `thresholdC` | 15 | Subtracted from the local weighted mean. Increasing it suppresses background but may lose faint strokes. |
| `maxSkewAngle` | 12° | Correction search range. Assumes an approximately upright document. |
| `angleStep` | 0.5° | Coarse skew search step; fine search uses one fifth of this step. |
| `maxWidth` in `estimateSkewAngle()` | 1000 | Working-copy width limit. Raise it if downsampling loses thin strokes. |
| `minComponentArea` | 3 pixels | Rejects tiny components. Raising it can remove punctuation and dots. |
| `minRowPixels` | 3 | Minimum active-row count. |
| `rowThresholdRatio` | 0.05 | Active-row threshold as a fraction of the peak. Lower it for short/faint lines; raise it for sparse inter-line valleys. Setting it to 0 uses only `minRowPixels`. |
| `maxGap` | 5 rows | Weak rows bridged within a line. Too large merges nearby lines; too small splits disconnected parts. |
| `minLineHeight` | 8 pixels | Rejects very short bands. Lower it for small writing. |
| `minLinePixels` | 40 pixels | Rejects sparse bands. Lower it for very short text. |
| `componentGap` | 8 rows | Distance for attaching detached components to a line. Too large attaches unrelated marks. |
| `cropMargin` | 3 pixels | Margin around full component bounds. |
| `targetHeight`, `padding` | 32, 2 pixels | Final line height and white padding; height must exceed twice the padding. |
| Gaussian kernel | 3×3 | In `removeNoise()`. Keep smoothing small to preserve thin strokes. |
| Closing kernel | 3×1 | In `closeSmallGaps()`; enabled only with `--close`. |

## Validation

Run the C++ checks:

```bash
ctest --test-dir build --output-on-failure
```

The checks cover projection counts, small-component removal, binary polarity, blank inputs, short gaps, document borders, detached dots, descenders, overlapping component heights, connected-line merging, grayscale normalization, padding, variable widths, rotation canvas coverage, and skew direction. Synthetic pages rotated by −7° and +7° must recover three separate lines after correction. Synthetic fixtures are written under `build/validation/` when the checks run. These are generated test drawings, not real handwriting.

Two real handwriting samples from [HTRPipeline's sample data](https://github.com/githubharald/HTRPipeline/tree/master/data) were also tested locally with OpenCV 4.13 on macOS. Both adaptive and Otsu pipelines detected 5 lines in `sample_1.png` and 11 in `sample_2.png`; rectangles and normalized crops were inspected. All exported images were checked for single-channel grayscale, 32-pixel height, white padding, and variable width. The first sample also retained five lines after artificial rotations of ±7°, and after JPEG conversion with optional closing enabled. Blank-page reruns removed stale numbered exports while retaining unrelated files; missing inputs and malformed arguments returned errors.

The first sample initially merged two lines, which led to the peak-relative valley threshold and the overlapping-height regression check. Visual inspection also caught detached punctuation, leading to attachment after full letter heights are restored. This is limited sample validation, not an accuracy benchmark or a guarantee for other writers.

The external samples are not redistributed in this repository. To repeat the real-image checks, download them into the ignored build directory:

```bash
mkdir -p build/validation
curl -fL https://raw.githubusercontent.com/githubharald/HTRPipeline/master/data/sample_1.png \
  -o build/validation/sample_1.png
curl -fL https://raw.githubusercontent.com/githubharald/HTRPipeline/master/data/sample_2.png \
  -o build/validation/sample_2.png
./build/htr_preprocess --input build/validation/sample_1.png --output_dir output
./build/htr_preprocess --input build/validation/sample_2.png --output_dir build/validation/sample_2_output
```

Tests can be excluded from a build with `-DBUILD_TESTING=OFF`.
