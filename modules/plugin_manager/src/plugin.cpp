#include "plugin.hpp"

#include <sstream>

cv::Plugin::Plugin(const cv::PluginInfo& info, const std::string& libPath) :
    info_(info), libPath_(libPath)
{
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

void* cv::Plugin::getSymbol(const std::string& name)
{
    if (!load())
    {
        std::ostringstream msg;
        msg << "Can't load plugin - " << info_.name;
        throw std::runtime_error(msg.str());
    }

    return lib_.getSymbol(name);
}
