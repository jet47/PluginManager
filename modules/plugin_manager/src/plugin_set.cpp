#include "plugin_set.hpp"

#include <sstream>

void cv::PluginSet::add(const Poco::SharedPtr<cv::Plugin>& plugin)
{
    plugins_.push_back(plugin);
}

Poco::SharedPtr<cv::Plugin> cv::PluginSet::getPlugin()
{
    if (plugins_.empty())
    {
        std::ostringstream msg;
        msg << "Can't find plugin for interface : " << interface_;
        throw std::runtime_error(msg.str());
    }

    for (size_t i = 0; i < plugins_.size(); ++i)
    {
        if (plugins_[i]->isLoaded())
            return plugins_[i];
    }

    Poco::SharedPtr<cv::Plugin> plugin;
    for (size_t i = 0; i < plugins_.size(); ++i)
    {
        if (plugins_[i]->load())
        {
            plugin = plugins_[i];
            break;
        }
    }

    if (plugin.isNull())
    {
        std::ostringstream msg;
        msg << "Can't find plugin for interface : " << interface_;
        throw std::runtime_error(msg.str());
    }

    return plugin;
}
