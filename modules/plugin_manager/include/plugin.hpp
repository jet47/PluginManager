#pragma once

#ifndef __OPENCV_PLUGIN_HPP__
#define __OPENCV_PLUGIN_HPP__

#include <string>

#include <Poco/SharedLibrary.h>

#include "plugin_info.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT Plugin
    {
    public:
        Plugin(const PluginInfo& info, const std::string& libPath);

        const PluginInfo& info() const { return info_; }

        bool isLoaded() const;
        bool load();
        void unload();

        void* getSymbol(const std::string& name);

    private:
        Plugin(const Plugin&);
        Plugin& operator =(const Plugin&);

        PluginInfo info_;
        std::string libPath_;
        Poco::SharedLibrary lib_;
    };
}

#endif // __OPENCV_MANAGER_HPP__
