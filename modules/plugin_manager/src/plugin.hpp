#pragma once

#ifndef __OPENCV_PLUGIN_HPP__
#define __OPENCV_PLUGIN_HPP__

#include <string>

#include "plugin_manager.hpp"
#include "utility.hpp"
#include "opencv_export.h"

namespace cv
{
    class OPENCV_NO_EXPORT Plugin : public PluginBase
    {
    public:
        static bool check(const std::string& fileName);

        Plugin(const PluginInfo& info, const std::string& libPath);

        PluginInfo info() const;
        std::string libPath() const;
        bool isLoaded() const;

        void load();
        void unload();

        AutoPtr<RefCountedObject> create(const std::string& interface, const ParameterMap& params);

    private:
        Plugin(const Plugin&);
        Plugin& operator =(const Plugin&);

        PluginInfo info_;
        std::string libPath_;
        SharedLibrary lib_;
    };
}

#endif // __OPENCV_MANAGER_HPP__
