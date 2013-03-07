#pragma once

#ifndef __OPENCV_PLUGIN_MANAGER_HPP__
#define __OPENCV_PLUGIN_MANAGER_HPP__

#include <string>
#include <vector>
#include <map>

#include <Poco/SingletonHolder.h>
#include <Poco/SharedPtr.h>
#include <Poco/File.h>

#include "plugin.hpp"
#include "plugin_set.hpp"
#include "opencv_export.h"

namespace cv
{
    class PluginManager;

    class OPENCV_EXPORT PluginManager
    {
    public:
        static PluginManager& instance();

        cv::PluginSet& getPluginSet(const std::string& interface);

    private:
        PluginManager();
        ~PluginManager();

        void reloadPluginsInfo();
        void processFolder(const Poco::File& folder);

        std::map<std::string, cv::PluginSet> plugins_;

        friend class Poco::SingletonHolder<cv::PluginManager>;
    };
}

#endif // __OPENCV_PLUGIN_MANAGER_HPP__
