#pragma once

#ifndef __OPENCV_PLUGIN_MANAGER_HPP__
#define __OPENCV_PLUGIN_MANAGER_HPP__

#include <vector>
#include <string>
#include <map>
#include <sstream>

#include "utility.hpp"

namespace cv
{
    struct PluginInfo
    {
        std::string name;
        std::string vendor;
        std::string version;
        std::vector<std::string> interfaces;
    };

    class OPENCV_EXPORT PluginBase : public RefCountedObject
    {
    public:
        virtual PluginInfo info() const = 0;
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
        mutable std::map<std::string, std::string> map_;
    };

    class OPENCV_EXPORT PluginManagerBase : public RefCountedObject
    {
    public:
        template <class Base>
        AutoPtr<Base> create(const std::string& interface, const ParameterMap& params = ParameterMap());

    protected:
        virtual AutoPtr<RefCountedObject> createImpl(const std::string& interface, const ParameterMap& params) = 0;
    };

    class OPENCV_EXPORT PluginManager : public PluginManagerBase
    {
    public:
        virtual void addPlugin(const std::string& libPath) = 0;
        virtual void addPluginDir(const std::string& dir, bool recursive = true) = 0;

        virtual void getPluginList(std::vector<AutoPtr<PluginBase> >& plugins) = 0;

        virtual void setPriority(const std::string& interfaceExpr, const std::string& pluginNameExpr, int priority) = 0;

        virtual bool hasPlugin(const std::string& interface) = 0;
    };

    OPENCV_EXPORT PluginManager& thePluginManager();
}

// ParameterMap

template <typename T>
void cv::ParameterMap::set(const std::string& name, T val)
{
    std::ostringstream ss;
    ss << val;
    map_[name] = ss.str();
}

template <typename T>
T cv::ParameterMap::get(const std::string& name) const
{
    std::istringstream ss(map_[name]);
    T val;
    ss >> val;
    return val;
}

// PluginManagerBase

template <class Base>
cv::AutoPtr<Base> cv::PluginManagerBase::create(const std::string& interface, const cv::ParameterMap& params)
{
    cv::AutoPtr<cv::RefCountedObject> obj = createImpl(interface, params);
    return obj.cast<Base>();
}

// macros

#if defined(_WIN32)
    #define OPENCV_PLUGIN_API __declspec(dllexport)
#else
    #define OPENCV_PLUGIN_API
#endif

#define OPENCV_BEGIN_PLUGIN_DECLARATION(plugin_name) \
    extern "C" OPENCV_PLUGIN_API void ocvGetPluginInfo(cv::PluginInfo* info); \
    void ocvGetPluginInfo(cv::PluginInfo* info) \
    { \
        info->name = plugin_name;

#define OPENCV_PLUGIN_VENDOR(plugin_vendor) \
        info->vendor = plugin_vendor;

#define OPENCV_PLUGIN_VERSION(plugin_version) \
        info->version = plugin_version;

#define OPENCV_PLUGIN_INTERFACE(plugin_interface) \
        info->interfaces.push_back(plugin_interface);

#define OPENCV_END_PLUGIN_DECLARATION() \
    }

#endif // __OPENCV_PLUGIN_MANAGER_HPP__
