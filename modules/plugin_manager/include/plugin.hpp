#pragma once

#ifndef __OPENCV_PLUGIN_HPP__
#define __OPENCV_PLUGIN_HPP__

#include <string>
#include <vector>

#include <Poco/SharedLibrary.h>

#include "opencv_export.h"

namespace cv
{
    struct OPENCV_EXPORT PluginInfo
    {
        std::string name;
        std::string vendor;
        std::string version;
        std::vector<std::string> interfaces;
    };

    class OPENCV_EXPORT Plugin
    {
    public:
        Plugin(const PluginInfo& info, const std::string& libPath);

        bool isLoaded() const;
        bool load();
        void unload();

        void* getSymbol(const std::string& name);

        const PluginInfo& info() const { return info_; }

    private:
        Plugin(const Plugin&);
        Plugin& operator =(const Plugin&);

        PluginInfo info_;
        std::string libPath_;
        Poco::SharedLibrary lib_;
    };
}

#endif // __OPENCV_MANAGER_HPP__
