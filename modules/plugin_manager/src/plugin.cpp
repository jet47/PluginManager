#include "plugin.hpp"

#include <sstream>

cv::Plugin::Plugin(const cv::PluginInfo& info, const std::string& libPath) :
    info_(info), libPath_(libPath)
{
}

const cv::PluginInfo& cv::Plugin::info() const
{
    return info_;
}

const std::string& cv::Plugin::libPath() const
{
    return libPath_;
}

bool cv::Plugin::isLoaded() const
{
    return lib_.isLoaded();
}

bool cv::Plugin::check(const std::string& fileName)
{
    try
    {
        cv::SharedLibrary lib(fileName);

        const bool hasInfo = lib.hasSymbol("ocvGetPluginInfo");
        const bool hasCreate = lib.hasSymbol("ocvCreatePlugin");

        return hasInfo && hasCreate;
    }
    catch(...)
    {
        return false;
    }
}

void cv::Plugin::load()
{
    if (isLoaded())
        return;

    lib_.load(libPath_);
}

void cv::Plugin::unload()
{
    if (isLoaded())
        lib_.unload();
}

namespace
{
    typedef cv::RefCountedObject* (*ocvCreatePlugin_t)(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger);
}

cv::AutoPtr<cv::RefCountedObject> cv::Plugin::create(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger)
{
    load();

    const ocvCreatePlugin_t ocvCreatePlugin = (ocvCreatePlugin_t) lib_.getSymbol("ocvCreatePlugin");

    cv::RefCountedObject* obj = ocvCreatePlugin(interface, params, logger);

    if (obj)
        return cv::AutoPtr<cv::RefCountedObject>(obj);

    return cv::AutoPtr<cv::RefCountedObject>();
}
