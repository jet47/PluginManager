#include "plugin_manager.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <Poco/Environment.h>
#include <Poco/Path.h>
#include <Poco/File.h>
#include <Poco/SharedLibrary.h>

cv::PluginManager& cv::PluginManager::instance()
{
    static Poco::SingletonHolder<cv::PluginManager> holder;

    return *holder.get();
}

cv::PluginManager::PluginManager()
{
    reloadPluginsInfo();
}

cv::PluginManager::~PluginManager()
{
}

cv::PluginSet& cv::PluginManager::getPluginSet(const std::string& interface)
{
    plugins_[interface].interface_ = interface;
    return plugins_[interface];
}

void cv::PluginManager::reloadPluginsInfo()
{
    plugins_.clear();

    const Poco::Path pluginBaseDirPath(Poco::Environment::get("OPENCV_PLUGIN_DIR"));

    Poco::File pluginBaseDir(pluginBaseDirPath);

    if (pluginBaseDir.exists() && pluginBaseDir.isDirectory())
        processFolder(pluginBaseDir);
}

namespace
{
    typedef cv::PluginInfo (*ocvGetPluginInfo_t)();
}

void cv::PluginManager::processFolder(const Poco::File& folder)
{
    std::vector<Poco::File> files;
    folder.list(files);

    for (size_t i = 0; i < files.size(); ++i)
    {
        const Poco::File& file = files[i];

        if (file.isDirectory())
            processFolder(file);
        else if (file.isFile())
        {
            const std::string path = Poco::Path(file.path()).absolute().toString();

            try
            {
                Poco::SharedLibrary lib;

                lib.load(path);
                const ocvGetPluginInfo_t ocvGetPluginInfo = (ocvGetPluginInfo_t) lib.getSymbol("ocvGetPluginInfo");

                const cv::PluginInfo info = ocvGetPluginInfo();

                lib.unload();

                Poco::SharedPtr<cv::Plugin> plugin(new cv::Plugin(info, path));

                for (size_t j = 0; j < info.interfaces.size(); ++j)
                {
                    const std::string& interface = info.interfaces[j];

                    plugins_[interface].interface_ = interface;
                    plugins_[interface].add(plugin);
                }
            }
            catch(const Poco::Exception& e)
            {
                std::cerr << e.message() << std::endl;
            }
        }
    }
}
