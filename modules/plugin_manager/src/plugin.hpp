#pragma once

#ifndef __OPENCV_PLUGIN_HPP__
#define __OPENCV_PLUGIN_HPP__

#include <string>

#include <Poco/SharedLibrary.h>

#include "plugin_manager.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT Plugin : public cv::PluginBase
    {
    public:
        Plugin(const PluginInfo& info, const std::string& libPath);

        cv::PluginInfo info() const;
        std::string libPath() const;
        bool isLoaded() const;

        bool load();
        void unload();

        cv::Ptr<cv::Object> create(const std::string& interface, const cv::ParameterMap& params);

    private:
        Plugin(const Plugin&);
        Plugin& operator =(const Plugin&);

        PluginInfo info_;
        std::string libPath_;
        Poco::SharedLibrary lib_;
    };
}

#endif // __OPENCV_MANAGER_HPP__
