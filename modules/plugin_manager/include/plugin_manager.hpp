#pragma once

#ifndef __OPENCV_PLUGIN_MANAGER_HPP__
#define __OPENCV_PLUGIN_MANAGER_HPP__

#include <vector>
#include <string>
#include <map>

#include "utility.hpp"
#include "opencv_export.h"

namespace cv
{
    struct PluginInfo
    {
        std::string name;
        std::string vendor;
        std::string version;
        std::vector<std::string> interfaces;
    };

    class OPENCV_EXPORT PluginBase : public Object
    {
    public:
        virtual cv::PluginInfo info() const = 0;
        virtual std::string libPath() const = 0;
        virtual bool isLoaded() const = 0;
    };

    class OPENCV_EXPORT ParameterMap
    {
    public:
        template <typename T>
        void set(const std::string& name, T val);

        template <typename T>
        T get(const std::string& name) const;

        bool has(const std::string& name) const;

    private:
        mutable std::map<std::string, cv::Any> map_;
    };

    class OPENCV_EXPORT PluginManagerBase : public Object
    {
    public:
        template <class Base>
        cv::Ptr<Base> create(const std::string& interface, const cv::ParameterMap& params = cv::ParameterMap());

    protected:
        virtual cv::Ptr<cv::Object> createImpl(const std::string& interface, const cv::ParameterMap& params) = 0;
    };

    class OPENCV_EXPORT PluginManager : public PluginManagerBase
    {
    public:
        virtual void updateCache(const std::string& baseDir = "", const std::string& manifestFile = "") = 0;
        virtual void loadPluginCache(const std::string& baseDir = "", const std::string& manifestFile = "") = 0;

        virtual void getPluginList(std::vector<cv::Ptr<cv::PluginBase> >& plugins) = 0;
    };

    OPENCV_EXPORT cv::PluginManager& thePluginManager();
}

// ParameterMap

template <typename T>
void cv::ParameterMap::set(const std::string& name, T val)
{
    map_[name] = val;
}

template <typename T>
T cv::ParameterMap::get(const std::string& name) const
{
    return map_[name];
}

// PluginManagerBase

template <class Base>
cv::Ptr<Base> cv::PluginManagerBase::create(const std::string& interface, const cv::ParameterMap& params)
{
    cv::Ptr<cv::Object> obj = createImpl(interface, params);
    return obj.cast<Base>();
}

// macros

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

#endif // __OPENCV_PLUGIN_MANAGER_HPP__
