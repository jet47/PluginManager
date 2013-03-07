#pragma once

#ifndef __OPENCV_PLUGIN_SET_HPP__
#define __OPENCV_PLUGIN_SET_HPP__

#include <string>
#include <vector>

#include <Poco/SharedPtr.h>

#include "plugin.hpp"
#include "opencv_export.h"

namespace cv
{
    class PluginManager;

    class OPENCV_EXPORT PluginSet
    {
    public:
        const std::vector<Poco::SharedPtr<cv::Plugin> >& plugins() const { return plugins_; }

        void add(const Poco::SharedPtr<cv::Plugin>& plugin);

        Poco::SharedPtr<cv::Plugin> getPlugin();

    private:
        std::string interface_;
        std::vector<Poco::SharedPtr<cv::Plugin> > plugins_;

        friend class cv::PluginManager;
    };
}

#endif // __OPENCV_PLUGIN_SET_HPP__
