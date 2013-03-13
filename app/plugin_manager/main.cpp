#include <iostream>
#include <vector>

#include "plugin_manager.hpp"
#include "gpu_module.hpp"

int main()
{
    cv::PluginManager* manager = cv::thePluginManager();
    manager->setLogLevel(true);

    manager->init();

    if (manager->hasPlugin("gpu.module"))
    {
        try
        {
            // Load all gpu plugins
            cv::PluginManagerBase* gpuModule = cv::theGpuModule();
            gpuModule->init();
        }
        catch(...)
        {
        }
    }

    std::vector<cv::AutoPtr<cv::PluginBase> > plugins;
    manager->getPluginList(plugins);

    std::cout << "OpenCV Plugins: \n" << std::endl;
    for (size_t i = 0; i < plugins.size(); ++i)
    {
        const cv::AutoPtr<cv::PluginBase>& plugin = plugins[i];
        const cv::PluginInfo& info = plugin->info();

        std::cout << "\t " << info.name << std::endl;
        std::cout << "\t\t Version : " << info.version << std::endl;
        std::cout << "\t\t Vendor : " << info.vendor << std::endl;
        std::cout << "\t\t Full path : " << plugin->libPath() << std::endl;
        std::cout << "\t\t Interfaces : " << std::endl;

        for (size_t j = 0; j < info.interfaces.size(); ++j)
            std::cout << "\t\t\t " << info.interfaces[j] << std::endl;

        std::cout << std::endl;
    }
}
