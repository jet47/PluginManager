#pragma once

#ifndef __OPENCV_PLUGIN_LIBRARY_HPP__
#define __OPENCV_PLUGIN_LIBRARY_HPP__

#include "plugin_info.hpp"

#if defined(_WIN32)
    #define OPENCV_PLUGIN_API __declspec(dllexport)
#else
    #define OPENCV_PLUGIN_API
#endif

#define OPENCV_BEGIN_PLUGIN_DECLARATION(plugin_name) \
    extern "C" \
    { \
        OPENCV_PLUGIN_API cv::PluginInfo ocvGetPluginInfo(); \
        OPENCV_PLUGIN_API bool ocvLoadPlugin(); \
        OPENCV_PLUGIN_API void ocvUnloadPlugin(); \
    } \
    cv::PluginInfo ocvGetPluginInfo() \
    { \
        cv::PluginInfo info;\
        info.name = plugin_name;

#define OPENCV_PLUGIN_VENDOR(plugin_vendor) \
        info.vendor = plugin_vendor;

#define OPENCV_PLUGIN_VERSION(plugin_version) \
        info.version = plugin_version;

#define OPENCV_PLUGIN_INTERFACE(plugin_interface) \
        info.interfaces.push_back(plugin_interface);

#define OPENCV_END_PLUGIN_DECLARATION() \
        return info; \
    }

#endif // __OPENCV_MANAGER_HPP__
