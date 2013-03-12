#pragma once

#ifndef __OPENCV_CORE_HPP__
#define __OPENCV_CORE_HPP__

#include <cstddef>

#include "utility.hpp"

namespace cv
{
    class Algorithm : public RefCountedObject
    {
    };

    enum
    {
        CV_8U,
        CV_32F
    };

    class OPENCV_API GpuMat
    {
    public:
        GpuMat();
        ~GpuMat();

        void create(int rows, int cols, int depth, int channels);
        void release();

        int rows;
        int cols;
        int depth;
        int channels;
        void* data;
        size_t step;

    private:
        GpuMat(const GpuMat& other);
        GpuMat& operator =(const GpuMat& other);
    };

    OPENCV_API void add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst);
}

#endif // __OPENCV_CORE_HPP__
