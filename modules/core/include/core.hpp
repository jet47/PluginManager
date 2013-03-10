#pragma once

#ifndef __OPENCV_CORE_HPP__
#define __OPENCV_CORE_HPP__

#include <cstddef>

#include "utility.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT Algorithm : public cv::RefCountedObject
    {
    };

    enum Depth
    {
        d8U,
        d32F
    };

    class OPENCV_EXPORT GpuMat
    {
    public:
        GpuMat();
        ~GpuMat();

        void create(int rows, int cols, cv::Depth depth, int channels);
        void release();

        int rows;
        int cols;
        cv::Depth depth;
        int channels;
        void* data;
        size_t step;

    private:
        GpuMat(const cv::GpuMat& other);
        cv::GpuMat& operator =(const cv::GpuMat& other);
    };

    OPENCV_EXPORT void add(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);
}

#endif // __OPENCV_CORE_HPP__
