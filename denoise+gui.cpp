
#include <opencv2/opencv.hpp>
#include <QApplication>
#include <QWidget>
#include <QSlider>
#include <QVBoxLayout>
#include <QLabel>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

Mat original;
int neighborhoodSize = 5;
double factorRatio = 50.0;


// Compute the determinant of the covariance matrix of a neighborhood
double computeCovarianceDet(const Mat& patch) {
    Mat mean;
    Mat data;

    // Ensure the patch is continuous
    Mat continuousPatch = patch.clone();

    // Reshape and convert to CV_64F
    continuousPatch.reshape(1, continuousPatch.rows * continuousPatch.cols).convertTo(data, CV_64F);

    // Compute covariance matrix
    Mat cov;
    calcCovarMatrix(data, cov, mean, COVAR_NORMAL | COVAR_ROWS);
    cov /= (patch.rows * patch.cols);

    // Return determinant of the covariance matrix
    return determinant(cov);

    
    
    // Mat mean;
    // Mat data;
    // patch.reshape(1, patch.rows * patch.cols).convertTo(data, CV_64F);
    // calcCovarMatrix(data, data, mean, COVAR_NORMAL | COVAR_ROWS);
    // Mat cov;
    // calcCovarMatrix(data, cov, mean, COVAR_NORMAL | COVAR_ROWS);
    // cov /= (patch.rows * patch.cols);
    // return determinant(cov);
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

// Apply adaptive Gaussian blur based on covariance determinant
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
            // int kernelSize = std::max(3, static_cast<int>(factorRatio / (det + 1e-5)));
            if (kernelSize % 2 == 0) kernelSize += 1;
            double sigma = kernelSize / 3.0;

            int kpad = kernelSize / 2;
            Rect kernelRoi(x + pad - kpad, y + pad - kpad, kernelSize, kernelSize);
            kernelRoi &= Rect(0, 0, padded.cols, padded.rows);
            Mat region = padded(kernelRoi);

            Mat kernel = createGaussianKernel(kernelSize, sigma);
            Vec3d sum = Vec3d(0, 0, 0);
            for (int i = 0; i < kernel.rows; ++i) {
                for (int j = 0; j < kernel.cols; ++j) {
                    Vec3b pix = region.at<Vec3b>(i, j);
                    double k = kernel.at<double>(i, j);
                    sum += Vec3d(pix) * k;
                }
            }
            output.at<Vec3b>(y, x) = Vec3b(sum);
        }
    }
    return output;
}

// Update image with GUI
void updateImage(QLabel* label) {
    Mat denoised = adaptiveDenoise(original, neighborhoodSize, factorRatio);
    Mat rgb;
    cvtColor(denoised, rgb, COLOR_BGR2RGB);
    QImage qImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(qImage).scaled(label->size(), Qt::KeepAspectRatio));
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    original = imread("cat-denoise.jpg");
    if (original.empty()) {
        cerr << "Image not loaded!" << endl;
        return -1;
    }

    QWidget window;
    window.setWindowTitle("Adaptive Denoising with Covariance Determinant");
    QVBoxLayout *layout = new QVBoxLayout(&window);

    QLabel* imageLabel = new QLabel;
    imageLabel->setFixedSize(800, 600);
    layout->addWidget(imageLabel);

    QSlider* neighborhoodSlider = new QSlider(Qt::Horizontal);
    neighborhoodSlider->setRange(3, 21);
    neighborhoodSlider->setValue(neighborhoodSize);
    QLabel* neighborhoodLabel = new QLabel("Neighborhood Size: " + QString::number(neighborhoodSize));
    QObject::connect(neighborhoodSlider, &QSlider::valueChanged, [&](int val) {
        if (val % 2 == 0) val += 1;
        neighborhoodSize = val;
        neighborhoodLabel->setText("Neighborhood Size: " + QString::number(neighborhoodSize));
        updateImage(imageLabel);
    });

    QSlider* factorSlider = new QSlider(Qt::Horizontal);
    factorSlider->setRange(1, 200);
    factorSlider->setValue(static_cast<int>(factorRatio));
    QLabel* factorLabel = new QLabel("Factor Ratio: " + QString::number(factorRatio));
    QObject::connect(factorSlider, &QSlider::valueChanged, [&](int val) {
        factorRatio = val;
        factorLabel->setText("Factor Ratio: " + QString::number(factorRatio));
        updateImage(imageLabel);
    });

    layout->addWidget(neighborhoodLabel);
    layout->addWidget(neighborhoodSlider);
    layout->addWidget(factorLabel);
    layout->addWidget(factorSlider);

    updateImage(imageLabel);
    window.show();
    return app.exec();
}
