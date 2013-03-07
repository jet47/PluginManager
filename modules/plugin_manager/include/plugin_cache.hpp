#pragma once

#ifndef __OPENCV_PLUGIN_CACHE_HPP__
#define __OPENCV_PLUGIN_CACHE_HPP__

#include <map>
#include <string>

#include "plugin_set.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT PluginCache
    {
    public:
        static void update();

        void load();

        std::map<std::string, cv::PluginSet>& plugins() { return plugins_; }

    private:
        std::map<std::string, cv::PluginSet> plugins_;
    };
}

#endif // __OPENCV_PLUGIN_CACHE_HPP__
