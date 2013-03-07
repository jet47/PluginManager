#pragma once

#ifndef __OPENCV_CORE_HPP__
#define __OPENCV_CORE_HPP__

#include <cstddef>

#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT GpuMat
    {
    public:
        GpuMat();
        ~GpuMat();

        void create(size_t rows, size_t cols, int channels);
        void release();

    private:
        GpuMat(const GpuMat&);
        GpuMat& operator =(const GpuMat&);

        size_t rows_;
        size_t cols_;
        int channels_;
        void* data_;
        size_t step_;
    };
}

#endif // __OPENCV_CORE_HPP__