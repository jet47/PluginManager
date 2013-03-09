#pragma once

#ifndef __OPENCV_GPU_MODULE_HPP__
#define __OPENCV_GPU_MODULE_HPP__

#include "utility.hpp"
#include "plugin_manager.hpp"
#include "core.hpp"
#include "opencv_export.h"

namespace cv
{
    OPENCV_EXPORT cv::PluginManagerBase& theGpuModule();

    class OPENCV_EXPORT GpuBasic : public cv::Object
    {
    public:
        virtual void* malloc2D(size_t height, size_t width, size_t& step) = 0;
        virtual void free(void* ptr) = 0;
    };

    class OPENCV_EXPORT GpuArithmBinary : public cv::Algorithm
    {
    public:
        virtual void apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst) = 0;
    };
}

#endif // __OPENCV_GPU_MODULE_HPP__
