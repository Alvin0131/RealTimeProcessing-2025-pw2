#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QLabel>
#include <QHBoxLayout>
#include <omp.h>

using namespace cv;
using namespace std;

// === Parameters ===
int matrixType = 0; // Change this to select anaglyph type
int kernelSize = 9;
double sigma = 2.0;
Mat stereo;

// === Create Gaussian Kernel ===
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

// === Manual Gaussian Blur ===
Mat manualGaussianBlur(const Mat& input, int kernelSize, double sigma) {
    int pad = kernelSize / 2;
    Mat kernel = createGaussianKernel(kernelSize, sigma);
    Mat padded;
    copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REPLICATE);
    Mat output = input.clone();


            for (int y = 0; y < input.rows; ++y) {
                for (int x = 0; x < input.cols; ++x) {
                    for (int c = 0; c < input.channels(); ++c) {
                        double sum = 0.0;
                        for (int ky = 0; ky < kernelSize; ++ky) {
                            for (int kx = 0; kx < kernelSize; ++kx) {
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

// === Apply Anaglyph Matrix ===
void applyMatrix(const Mat& img, int matrixType, Mat& leftTransformed, Mat& rightTransformed) {
    Mat leftMatrix, rightMatrix;
    if (matrixType == 0) { // Color
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0, 0, 1);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    } else if (matrixType == 1) { // Half-Color
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    } else if (matrixType == 2) { // Optimized
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0, 0.7, 0.3);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    } else if (matrixType == 3) { // Gray
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0, 0, 0);
    } else { // True
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0);
    }

    Mat leftHalf = img(Rect(0, 0, img.cols / 2, img.rows));
    Mat rightHalf = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));
    transform(leftHalf, leftTransformed, leftMatrix);
    transform(rightHalf, rightTransformed, rightMatrix);
}

void updateImage(QLabel* label) {
    Mat leftTransformed, rightTransformed;
    applyMatrix(stereo, matrixType, leftTransformed, rightTransformed);

    Mat leftBlurred = manualGaussianBlur(leftTransformed, kernelSize, sigma);
    Mat rightBlurred = manualGaussianBlur(rightTransformed, kernelSize, sigma);

    Mat finalImage;
    add(leftBlurred, rightBlurred, finalImage);

    Mat rgb;
    cvtColor(finalImage, rgb, COLOR_BGR2RGB);
    QImage qImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(qImage).scaled(label->size(), Qt::KeepAspectRatio));
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    stereo = imread("Macrophant3D_3.jpg");
    if (stereo.empty()) {
        cerr << "Image not loaded!" << endl;
        return -1;
    }

    QWidget window;
    window.setWindowTitle("Anaglyph + Gaussian GUI");
    QVBoxLayout *mainLayout = new QVBoxLayout(&window);

    QLabel* imageLabel = new QLabel;
    imageLabel->setFixedSize(800, 600);
    mainLayout->addWidget(imageLabel);

    QHBoxLayout* buttonLayout = new QHBoxLayout;
    QStringList names = {"Color", "Half-Color", "Optimized", "Gray", "True"};
    for (int i = 0; i < names.size(); ++i) {
        QPushButton* btn = new QPushButton(names[i]);
        buttonLayout->addWidget(btn);
        QObject::connect(btn, &QPushButton::clicked, [=]() {
            matrixType = i;
            updateImage(imageLabel);
        });
    }
    mainLayout->addLayout(buttonLayout);

    QSlider* kernelSlider = new QSlider(Qt::Horizontal);
    kernelSlider->setRange(1, 31);
    kernelSlider->setValue(kernelSize);
    QLabel* kernelLabel = new QLabel("Kernel Size: " + QString::number(kernelSize));
    QObject::connect(kernelSlider, &QSlider::valueChanged, [&](int val) {
        kernelSize = (val % 2 == 0) ? val + 1 : val;
        kernelLabel->setText("Kernel Size: " + QString::number(kernelSize));
        updateImage(imageLabel);
    });

    QSlider* sigmaSlider = new QSlider(Qt::Horizontal);
    sigmaSlider->setRange(1, 100);
    sigmaSlider->setValue(static_cast<int>(sigma * 10));
    QLabel* sigmaLabel = new QLabel("Sigma: " + QString::number(sigma));
    QObject::connect(sigmaSlider, &QSlider::valueChanged, [&](int val) {
        sigma = val / 10.0;
        sigmaLabel->setText("Sigma: " + QString::number(sigma));
        updateImage(imageLabel);
    });

    mainLayout->addWidget(kernelLabel);
    mainLayout->addWidget(kernelSlider);
    mainLayout->addWidget(sigmaLabel);
    mainLayout->addWidget(sigmaSlider);

    updateImage(imageLabel);
    window.show();

    return app.exec();
}
