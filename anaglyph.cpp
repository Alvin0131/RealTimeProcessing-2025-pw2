#include <opencv2/opencv.hpp>
#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <iostream>

using namespace cv;
using namespace std;

Mat img, transformed;
int matrixType = 0;

// Function to apply transformation based on matrix type
void applyMatrix(const Mat& img, int matrixType, Mat& leftTransformed, Mat& rightTransformed) {
    Mat leftMatrix, rightMatrix;

    if (matrixType == 0) { // Color Anaglyph
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0, 0, 1);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    } 
    else if (matrixType == 1) { // Half-Color Anaglyph
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    }
    else if (matrixType == 2) { // Optimized Anaglyph
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0, 0.7, 0.3);
        rightMatrix = (Mat_<float>(3, 3) << 1, 0, 0,  0, 1, 0,  0, 0, 0);
    }
    else if (matrixType == 3) { // Gray Anaglyph
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0, 0, 0);
    }
    else { // True Anaglyph
        leftMatrix = (Mat_<float>(3, 3) << 0, 0, 0,  0, 0, 0,  0.299, 0.587, 0.114);
        rightMatrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114, 0,0,0, 0, 0, 0);
    }

    // Split the image into left and right halves
    Mat leftHalf = img(Rect(0, 0, img.cols / 2, img.rows));
    Mat rightHalf = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));

    // Apply transformations
    
    transform(leftHalf, leftTransformed, leftMatrix);
    transform(rightHalf, rightTransformed, rightMatrix);

    
}

// Function to update the image when a button is pressed
void updateImage(int type) {
    matrixType = type;
    Mat leftTransformed, rightTransformed;
    applyMatrix(img, matrixType, leftTransformed, rightTransformed);

    // Merge both halves to create the final anaglyph image
    Mat newImage;
    add(leftTransformed, rightTransformed, newImage);

    // Show the images
    imshow("Anaglyph Image", newImage);
    imshow("Left Transformed Image", leftTransformed);
    imshow("Right Transformed Image", rightTransformed);

}

// Qt GUI Class
class AnaglyphWindow : public QWidget {
public:
    AnaglyphWindow(QWidget *parent = nullptr) : QWidget(parent) {
        QVBoxLayout *layout = new QVBoxLayout(this);

        QPushButton *btnColor = new QPushButton("Color Anaglyph", this);
        QPushButton *btnHalfColor = new QPushButton("Half-Color Anaglyph", this);
        QPushButton *btnOptimized = new QPushButton("Optimized Anaglyph", this);
        QPushButton *btnGray = new QPushButton("Gray Anaglyph", this);
        QPushButton *btnTrue = new QPushButton("True Anaglyph", this);

        layout->addWidget(btnColor);
        layout->addWidget(btnHalfColor);
        layout->addWidget(btnOptimized);
        layout->addWidget(btnGray);
        layout->addWidget(btnTrue);

        connect(btnColor, &QPushButton::clicked, this, []() { updateImage(0); });
        connect(btnHalfColor, &QPushButton::clicked, this, []() { updateImage(1); });
        connect(btnOptimized, &QPushButton::clicked, this, []() { updateImage(2); });
        connect(btnGray, &QPushButton::clicked, this, []() { updateImage(3); });
        connect(btnTrue, &QPushButton::clicked, this, []() { updateImage(4); });

        setLayout(layout);
        setWindowTitle("Anaglyph Selector");
        resize(300, 200);
    }
};

// Main Function
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    img = imread("Macrophant3D_3.jpg");
    if (img.empty()) {
        cerr << "Could not open image!" << endl;
        return -1;
    }

    // Show initial image
    imshow("Anaglyph Image", img);

    // Launch Qt GUI
    AnaglyphWindow window;
    window.show();

    return app.exec();
}
