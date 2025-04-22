QT       += core gui widgets

CONFIG   += console c++17
CONFIG   -= app_bundle

SOURCES  += qtguitest.cpp

INCLUDEPATH += /opt/homebrew/opt/qt/include
INCLUDEPATH += /opt/homebrew/include/opencv4

LIBS += -L/opt/homebrew/lib \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -F/opt/homebrew/opt/qt/lib \
        -framework QtCore \
        -framework QtGui \
        -framework QtWidgets

# Use LLVM's clang++ for OpenMP
QMAKE_CC = /opt/homebrew/opt/llvm/bin/clang
QMAKE_CXX = /opt/homebrew/opt/llvm/bin/clang++
QMAKE_LINK = /opt/homebrew/opt/llvm/bin/clang++

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp