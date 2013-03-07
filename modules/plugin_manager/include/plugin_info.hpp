#pragma once

#ifndef __OPENCV_PLUGIN_INFO_HPP__
#define __OPENCV_PLUGIN_INFO_HPP__

#include <string>
#include <vector>

namespace cv
{
    struct PluginInfo
    {
        std::string name;
        std::string vendor;
        std::string version;
        std::vector<std::string> interfaces;
    };
}

#endif // __OPENCV_PLUGIN_INFO_HPP__
