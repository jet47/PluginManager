#include "plugin.hpp"

cv::Plugin::Plugin(const cv::PluginInfo& info, const std::string& libPath) :
    info_(info), libPath_(libPath)
{
}

bool cv::Plugin::isLoaded() const
{
    return lib_.isLoaded();
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

void* cv::Plugin::getSymbol(const std::string& name)
{
    load();

    return lib_.getSymbol(name);
}
