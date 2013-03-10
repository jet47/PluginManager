#include "plugin_manager.hpp"

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <iterator>

#include "plugin.hpp"

bool cv::ParameterMap::has(const std::string& name) const
{
    return map_.find(name) != map_.end();
}

namespace
{
    class PluginManagerImpl : public cv::PluginManager
    {
    public:
        void addPlugin(const std::string& libPath);
        void addPluginDir(const std::string& dir, bool recursive = true);

        void getPluginList(std::vector<cv::AutoPtr<cv::PluginBase> >& plugins);

        void setPriority(const std::string& interfaceExpr, const std::string& pluginNameExpr, int priority);

        bool hasPlugin(const std::string& interface);

    protected:
        cv::AutoPtr<cv::RefCountedObject> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        std::vector<cv::AutoPtr<cv::Plugin> > allPlugins_;
        std::map<std::string, std::vector<cv::AutoPtr<cv::Plugin> > > pluginsMap_;

        struct PrioritySettings
        {
            std::string interfaceExpr;
            std::string pluginNameExpr;
            int priority;
        };
        std::vector<PrioritySettings> prioritySettings_;
    };

    typedef void (*ocvGetPluginInfo_t)(cv::PluginInfo* info);

    void PluginManagerImpl::addPlugin(const std::string& libPath)
    {
        if (!cv::Plugin::check(libPath))
            throw std::runtime_error("Incorrect plugin");

        cv::SharedLibrary lib;

        lib.load(libPath);
        const ocvGetPluginInfo_t ocvGetPluginInfo = (ocvGetPluginInfo_t) lib.getSymbol("ocvGetPluginInfo");

        cv::PluginInfo info;
        ocvGetPluginInfo(&info);

        lib.unload();

        cv::AutoPtr<cv::Plugin> plugin = new cv::Plugin(info, libPath);
        allPlugins_.push_back(plugin);

        for (size_t i = 0; i < info.interfaces.size(); ++i)
            pluginsMap_[info.interfaces[i]].push_back(plugin);
    }

    void PluginManagerImpl::addPluginDir(const std::string& dir, bool recursive)
    {
        std::vector<std::string> files;
        cv::Path::glob(dir + "/*" + cv::SharedLibrary::suffix(), files, recursive);

        for (size_t i = 0; i < files.size(); ++i)
        {
            if (!cv::Plugin::check(files[i]))
                addPlugin(files[i]);
        }
    }

    void PluginManagerImpl::getPluginList(std::vector<cv::AutoPtr<cv::PluginBase> >& plugins)
    {
        if (allPlugins_.empty())
            addPluginDir(cv::Environment::get("OPENCV_PLUGIN_DIR", "."));

        plugins.clear();

        std::copy(allPlugins_.begin(), allPlugins_.end(), std::back_inserter(plugins));
    }

    void PluginManagerImpl::setPriority(const std::string& interfaceExpr, const std::string& pluginNameExpr, int priority)
    {
        PrioritySettings s;
        s.interfaceExpr = interfaceExpr;
        s.pluginNameExpr = pluginNameExpr;
        s.priority = priority;
        prioritySettings_.push_back(s);
    }

    struct PluginCompare
    {
        mutable std::map<std::string, int> priorityMap;

        bool operator ()(const cv::AutoPtr<cv::Plugin>& pluginA, const cv::AutoPtr<cv::Plugin>& pluginB) const
        {
            // TODO : regexp
            return priorityMap[pluginA->info().name] > priorityMap[pluginB->info().name];
        }
    };

    cv::AutoPtr<cv::RefCountedObject> PluginManagerImpl::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        if (allPlugins_.empty())
            addPluginDir(cv::Environment::get("OPENCV_PLUGIN_DIR", "."));

        std::vector<cv::AutoPtr<cv::Plugin> > plugins = pluginsMap_[interface];

        if (plugins.empty())
        {
            std::ostringstream msg;
            msg << "Can't find plugin for interface : " << interface;
            throw std::runtime_error(msg.str());
        }

        PluginCompare pluginCmp;
        for (size_t i = 0; i < plugins.size(); ++i)
        {
            const cv::AutoPtr<cv::Plugin>& plugin = plugins[i];
            pluginCmp.priorityMap[plugin->info().name] = plugin->isLoaded();
        }
        for (size_t i = 0; i < prioritySettings_.size(); ++i)
        {
            const PrioritySettings& s = prioritySettings_[i];

            // TODO : regex
            if (s.interfaceExpr == interface)
                pluginCmp.priorityMap[s.pluginNameExpr] = s.priority;
        }

        std::sort(plugins.begin(), plugins.end(), pluginCmp);

        for (size_t i = 0; i < plugins.size(); ++i)
        {
            cv::AutoPtr<cv::Plugin>& plugin = plugins[i];

            // TODO : regex
            if (pluginCmp.priorityMap[plugin->info().name] < 0)
                break;

            bool wasLoaded = plugin->isLoaded();

            if (!wasLoaded)
                plugin->load();

            cv::AutoPtr<cv::RefCountedObject> obj = plugin->create(interface, params);
            if (!obj.isNull())
                return obj;

            if (!wasLoaded)
                plugin->unload();
        }

        std::ostringstream msg;
        msg << "Can't find plugin for interface : " << interface;
        throw std::runtime_error(msg.str());
    }

    bool PluginManagerImpl::hasPlugin(const std::string& interface)
    {
        return pluginsMap_.find(interface) != pluginsMap_.end();
    }
}

cv::PluginManager& cv::thePluginManager()
{
    static cv::SingletonHolder<PluginManagerImpl> holder;
    return *holder.get();
}
