#pragma once

#ifndef __OPENCV_GPU_MODULE_HPP__
#define __OPENCV_GPU_MODULE_HPP__

#include <Poco/SingletonHolder.h>
#include <Poco/SharedPtr.h>

#include "plugin.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT GpuModuleManager
    {
    public:
        virtual ~GpuModuleManager() {}

        virtual Poco::SharedPtr<cv::Plugin> getPlugin(const std::string& name) = 0;
    };

    class OPENCV_EXPORT GpuModule
    {
    public:
        static GpuModule& instance();

        Poco::SharedPtr<cv::Plugin> getPlugin(const std::string& name);

    private:
        GpuModule();
        ~GpuModule();

        Poco::SharedPtr<cv::Plugin> gpuPlugin_;
        Poco::SharedPtr<cv::GpuModuleManager> gpuModuleManager_;

        friend class Poco::SingletonHolder<cv::GpuModule>;
    };
}

#endif // __OPENCV_GPU_MODULE_HPP__
