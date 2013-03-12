#pragma once

#ifndef __OPENCV_GPU_MODULE_HPP__
#define __OPENCV_GPU_MODULE_HPP__

#include "utility.hpp"
#include "plugin_manager.hpp"
#include "core.hpp"

namespace cv
{
    OPENCV_API PluginManagerBase* theGpuModule();

    class GpuBasic : public RefCountedObject
    {
    public:
        virtual void* malloc2D(size_t height, size_t width, size_t& step) = 0;
        virtual void free(void* ptr) = 0;
    };

    class GpuArithmBinary : public Algorithm
    {
    public:
        virtual void apply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) = 0;
    };
}

#endif // __OPENCV_GPU_MODULE_HPP__
