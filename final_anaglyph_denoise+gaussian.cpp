#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace std;

// Function to compute the determinant of the covariance matrix of a patch
double computeCovarianceDet(const Mat& patch) {
    Mat continuousPatch = patch.clone();
    Mat data;
    continuousPatch.reshape(1, patch.rows * patch.cols).convertTo(data, CV_64F);

    Mat mean, cov;
    calcCovarMatrix(data, cov, mean, COVAR_NORMAL | COVAR_ROWS);
    cov /= (patch.rows * patch.cols);

    return determinant(cov);
}

// Adaptive denoising using covariance determinant
Mat adaptiveDenoise(const Mat& input, int neighborhoodSize, double factorRatio) {
    int pad = neighborhoodSize / 2;
    Mat padded;
    copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REFLECT);
    Mat output = input.clone();

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            Rect roi(x, y, neighborhoodSize, neighborhoodSize);
            roi &= Rect(0, 0, padded.cols, padded.rows);
            Mat patch = padded(roi);

            double det = computeCovarianceDet(patch);
            int kernelSize = std::max(3, std::min(21, static_cast<int>(factorRatio / (det + 1e-5))));
            if (kernelSize % 2 == 0) kernelSize += 1;

            int kpad = kernelSize / 2;
            Rect kernelRoi(x + pad - kpad, y + pad - kpad, kernelSize, kernelSize);
            kernelRoi &= Rect(0, 0, padded.cols, padded.rows);
            Mat region = padded(kernelRoi);

            Mat blurred;
            GaussianBlur(region, blurred, Size(kernelSize, kernelSize), 0);

            int centerY = y + pad - kernelRoi.y;
            int centerX = x + pad - kernelRoi.x;
            output.at<Vec3b>(y, x) = blurred.at<Vec3b>(centerY, centerX);
        }
    }

    return output;
}

// Create Gaussian kernel
Mat createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    Mat kernel(ksize, ksize, CV_64F);
    double sum = 0.0;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            double value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + half, j + half) = value;
            sum += value;
        }
    }
    return kernel / sum;
}

// Manual Gaussian blur
Mat manualGaussianBlur(const Mat& input, const Mat& kernel) {
    int pad = kernel.rows / 2;
    Mat padded;
    copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REPLICATE);
    Mat output = input.clone();


    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int c = 0; c < input.channels(); ++c) {
                double sum = 0.0;
                for (int ky = 0; ky < kernel.rows; ++ky) {
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        int px = x + kx;
                        int py = y + ky;
                        double k = kernel.at<double>(ky, kx);
                        sum += k * padded.at<Vec3b>(py, px)[c];
                    }
                }
                output.at<Vec3b>(y, x)[c] = static_cast<uchar>(sum);
            }
        }
    }
    return output;
}

// Apply anaglyph matrix
void applyMatrix(const Mat& img, int matrixType, Mat& leftTransformed, Mat& rightTransformed) {
    Mat leftMatrix, rightMatrix;
    if (matrixType == 0) { // Color
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 1);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
    }
    else if (matrixType == 1) { // Half-Color
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
    }
    else if (matrixType == 2) { // Optimized
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0.7, 0.3);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0);
    }
    else if (matrixType == 3) { // Gray
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0, 0, 0);
    }
    else { // True
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0);
    }

    Mat leftHalf = img(Rect(0, 0, img.cols / 2, img.rows));
    Mat rightHalf = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));
    transform(leftHalf, leftTransformed, leftMatrix);
    transform(rightHalf, rightTransformed, rightMatrix);
}

int main() {
    

    string anachoice;
    cout << "Choose anaglyph type (0: Color, 1: Half-Color, 2: Optimized, 3: Gray, 4: True): ";
    cin >> anachoice;
    int matrixType = stoi(anachoice);
    if (matrixType < 0 || matrixType > 4) {
        cerr << "Invalid anaglyph type." << endl;
        return -1;
    }
    if (matrixType == 0) {
        cout << "Color anaglyph selected." << endl;
    } else if (matrixType == 1) {
        cout << "Half-Color anaglyph selected." << endl;
    } else if (matrixType == 2) {
        cout << "Optimized anaglyph selected." << endl;
    } else if (matrixType == 3) {
        cout << "Gray anaglyph selected." << endl;
    } else {
        cout << "True anaglyph selected." << endl;
    }
    string choice;
    cout << "Choose processing type (denoise/gaussian): ";
    cin >> choice;
    if (choice != "denoise" && choice != "gaussian") {
        cerr << "Invalid choice." << endl;
        return -1;
    }

    if (choice == "denoise") {
        // Load image for denoising
        Mat stereo = imread("Macrophant3D_3.jpg");
        if (stereo.empty()) {
            cerr << "Error: Could not load image for denoising." << endl;
            return -1;
        }
        Mat leftTransformed, rightTransformed;
        applyMatrix(stereo, matrixType, leftTransformed, rightTransformed);


        auto start = chrono::high_resolution_clock::now();
        
        Mat leftBlurred = adaptiveDenoise(leftTransformed, 3, 20.0);
        Mat rightBlurred = adaptiveDenoise(rightTransformed, 3, 20.0);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Denoising Time: " << duration.count() << " seconds" << endl;
        Mat finalImage;
        add(leftBlurred, rightBlurred, finalImage);

        imshow("Denoised Image", finalImage);
        
    } else if (choice == "gaussian") {
        // Load image for anaglyph processing
        Mat stereo = imread("Macrophant3D_3.jpg");
        if (stereo.empty()) {
            cerr << "Error: Could not load image for gaussian." << endl;
            return -1;
        }

        Mat leftTransformed, rightTransformed;
        applyMatrix(stereo, matrixType, leftTransformed, rightTransformed);
        
        auto start = chrono::high_resolution_clock::now();

        Mat kernel = createGaussianKernel(11, 7.0);
        Mat leftBlurred = manualGaussianBlur(leftTransformed, kernel);
        Mat rightBlurred = manualGaussianBlur(rightTransformed, kernel);
       
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Gaussian Time: " << duration.count() << " seconds" << endl;
        
        Mat finalImage;
        add(leftBlurred, rightBlurred, finalImage);

        imshow("Gaussian Image", finalImage);
    } else {
        cerr << "Invalid choice." << endl;
        return -1;
    }

    waitKey(0);
    return 0;
}
