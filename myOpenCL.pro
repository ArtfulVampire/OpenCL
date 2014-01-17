#-------------------------------------------------
#
# Project created by QtCreator 2014-01-13T18:12:22
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = myOpenCL
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


LIBS += -L/opt/intel/opencl-1.2-3.2.1.16712/lib64 -lOpenCL
LIBS += -L/usr/lib/x86_64-linux-gnu


SOURCES += main.cpp

OTHER_FILES += \
    kernel.cl
