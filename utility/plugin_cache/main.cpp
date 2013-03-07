#include <iostream>

#include "plugin_cache.hpp"

int main()
{
    cv::PluginCache cache;
    cache.load();

    typedef std::map<std::string, cv::PluginSet> PluginMap;

    PluginMap& pluginSets = cache.plugins();

    for (PluginMap::const_iterator it = pluginSets.begin(); it != pluginSets.end(); ++it)
    {
        std::cout << "Interface : " << it->first << std::endl;

        const std::vector<Poco::SharedPtr<cv::Plugin> >& plugins = it->second.plugins();

        for (size_t i = 0; i < plugins.size(); ++i)
        {
            const Poco::SharedPtr<cv::Plugin>& plugin = plugins[i];
            const cv::PluginInfo& info = plugin->info();

            std::cout << "\t Plugin : " << info.name << std::endl;
            std::cout << "\t Vendor : " << info.vendor << std::endl;
            std::cout << "\t Version : " << info.version << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;
}
