#include "plugin.hpp"

#include <sstream>

cv::Plugin::Plugin(const cv::PluginInfo& info, const std::string& libPath) :
    info_(info), libPath_(libPath)
{
}

cv::PluginInfo cv::Plugin::info() const
{
    return info_;
}

std::string cv::Plugin::libPath() const
{
    return libPath_;
}

bool cv::Plugin::isLoaded() const
{
    return lib_.isLoaded();
}

namespace
{
    typedef bool (*ocvLoadPlugin_t)();
}

bool cv::Plugin::load()
{
    if (isLoaded())
        return true;

    lib_.load(libPath_);

    if (!lib_.hasSymbol("ocvLoadPlugin"))
        return true;
    else
    {
        const ocvLoadPlugin_t ocvLoadPlugin = (ocvLoadPlugin_t) lib_.getSymbol("ocvLoadPlugin");

        return ocvLoadPlugin();
    }
}

void cv::Plugin::unload()
{
    if (isLoaded())
        lib_.unload();
}

namespace
{
    typedef cv::Object* (*ocvPluginCreate_t)(const std::string& interface, const cv::ParameterMap& params);
}

cv::Ptr<cv::Object> cv::Plugin::create(const std::string& interface, const cv::ParameterMap& params)
{
    if (!load())
    {
        std::ostringstream msg;
        msg << "Can't load plugin - " << info_.name;
        throw std::runtime_error(msg.str());
    }

    if (!lib_.hasSymbol("ocvPluginCreate"))
    {
        std::ostringstream msg;
        msg << "Incorrect plugin - " << info_.name;
        throw std::runtime_error(msg.str());
    }

    const ocvPluginCreate_t ocvPluginCreate = (ocvPluginCreate_t) lib_.getSymbol("ocvPluginCreate");

    cv::Object* obj = ocvPluginCreate(interface, params);

    if (obj)
        return cv::Ptr<cv::Object>(obj);

    return cv::Ptr<cv::Object>();
}
