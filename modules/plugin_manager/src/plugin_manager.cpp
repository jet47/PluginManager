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

namespace
{
    class PluginLoggerImpl : public cv::PluginLogger
    {
    public:
        void message(const std::string& msg);

        std::string getOutput() const;

    private:
        std::ostringstream ostr_;
    };

    void PluginLoggerImpl::message(const std::string& msg)
    {
        ostr_ << '\t' << msg << std::endl;
    }

    std::string PluginLoggerImpl::getOutput() const
    {
        return ostr_.str();
    }

    class PluginManagerImpl : public cv::PluginManager
    {
    public:
        PluginManagerImpl();

        void addPlugin(const std::string& libPath);
        void addPluginDir(const std::string& dir, bool recursive = true);

        void getPluginList(std::vector<cv::AutoPtr<cv::PluginBase> >& plugins);

        void setPriority(const std::string& interfaceExpr, const std::string& pluginNameExpr, int priority);
        void clearPrioritySettings();

        bool hasPlugin(const std::string& interface);

        void setLogLevel(bool verbose);

    protected:
        cv::AutoPtr<cv::RefCountedObject> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        std::vector<cv::AutoPtr<cv::Plugin> > allPlugins_;
        std::map<std::string, std::vector<cv::Plugin*> > pluginsMap_;

        struct PrioritySettings
        {
            std::string interfaceExpr;
            std::string pluginNameExpr;
            int priority;
        };
        std::vector<PrioritySettings> prioritySettings_;

        cv::Mutex mutex_;

        bool verbose_;
    };

    PluginManagerImpl::PluginManagerImpl() : verbose_(false)
    {
    }

    typedef void (*ocvGetPluginInfo_t)(cv::PluginInfo* info);

    void PluginManagerImpl::addPlugin(const std::string& libPath)
    {
        if (!cv::Plugin::check(libPath))
        {
            std::ostringstream msg;
            msg << libPath << " is not a correct OpenCV plugin";
            throw std::runtime_error(msg.str());
        }

        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Try to load " << libPath << std::endl;

        cv::SharedLibrary lib;

        lib.load(libPath);
        const ocvGetPluginInfo_t ocvGetPluginInfo = (ocvGetPluginInfo_t) lib.getSymbol("ocvGetPluginInfo");

        cv::PluginInfo info;
        ocvGetPluginInfo(&info);

        lib.unload();

        cv::Mutex::ScopedLock lock(mutex_);

        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Add " << libPath << std::endl;

        cv::AutoPtr<cv::Plugin> plugin(new cv::Plugin(info, libPath));
        allPlugins_.push_back(plugin);

        for (size_t i = 0; i < info.interfaces.size(); ++i)
            pluginsMap_[info.interfaces[i]].push_back(plugin.get());
    }

    void PluginManagerImpl::addPluginDir(const std::string& dir, bool recursive)
    {
        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Scan " << dir << ' ' << (recursive ? "[Recursive]" : "[Non Recursive]") << std::endl;

        std::vector<std::string> files;
        cv::Path::glob(dir + "/*" + cv::SharedLibrary::suffix(), files, recursive);

        for (size_t i = 0; i < files.size(); ++i)
        {
            if (cv::Plugin::check(files[i]))
                addPlugin(files[i]);
        }
    }

    void PluginManagerImpl::getPluginList(std::vector<cv::AutoPtr<cv::PluginBase> >& plugins)
    {
        if (allPlugins_.empty())
            addPluginDir(cv::Environment::get("OPENCV_PLUGIN_DIR", "."));

        plugins.clear();

        cv::Mutex::ScopedLock lock(mutex_);

        std::copy(allPlugins_.begin(), allPlugins_.end(), std::back_inserter(plugins));
    }

    void PluginManagerImpl::setPriority(const std::string& interfaceExpr, const std::string& pluginNameExpr, int priority)
    {
        PrioritySettings s;
        s.interfaceExpr = interfaceExpr;
        s.pluginNameExpr = pluginNameExpr;
        s.priority = priority;

        cv::Mutex::ScopedLock lock(mutex_);

        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Add new priority settings : " << interfaceExpr << " " << pluginNameExpr << " " << priority << std::endl;

        prioritySettings_.push_back(s);
    }

    void PluginManagerImpl::clearPrioritySettings()
    {
        cv::Mutex::ScopedLock lock(mutex_);

        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Clear priority settings" << std::endl;

        prioritySettings_.clear();
    }

    struct PluginCompare
    {
        mutable std::map<std::string, int> priorityMap;

        bool operator ()(const cv::Plugin* pluginA, const cv::Plugin* pluginB) const
        {
            // TODO : regexp
            return priorityMap[pluginA->info().name] > priorityMap[pluginB->info().name];
        }
    };

    cv::AutoPtr<cv::RefCountedObject> PluginManagerImpl::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        if (allPlugins_.empty())
            addPluginDir(cv::Environment::get("OPENCV_PLUGIN_DIR", "."));

        if (verbose_)
            std::cout << "OpenCV Plugin Manager : Look for " << interface << std::endl;

        cv::Mutex::ScopedLock lock(mutex_);

        std::vector<cv::Plugin*> plugins = pluginsMap_[interface];

        if (plugins.empty())
        {
            std::ostringstream msg;
            msg << "Can't find plugin for interface : " << interface;
            throw std::runtime_error(msg.str());
        }

        PluginCompare pluginCmp;
        for (size_t i = 0; i < plugins.size(); ++i)
        {
            const cv::Plugin* plugin = plugins[i];
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
            cv::Plugin* plugin = plugins[i];

            // TODO : regex
            if (pluginCmp.priorityMap[plugin->info().name] < 0)
                break;

            bool wasLoaded = plugin->isLoaded();

            if (!wasLoaded)
                plugin->load();

            PluginLoggerImpl logger;

            cv::AutoPtr<cv::RefCountedObject> obj = plugin->create(interface, params, &logger);

            if (verbose_)
            {
                std::string pluginOutput = logger.getOutput();
                if (!pluginOutput.empty())
                {
                    std::cout << "OpenCV Plugin Manager : " << plugin->info().name << ':' << std::endl;
                    std::cout << pluginOutput;
                }
            }

            if (!obj.isNull())
            {
                if (verbose_)
                    std::cout << "OpenCV Plugin Manager : Use " << plugin->info().name << std::endl;

                return obj;
            }

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

    void PluginManagerImpl::setLogLevel(bool verbose)
    {
        verbose_ = verbose;
    }
}

cv::PluginManager *cv::thePluginManager()
{
    static cv::SingletonHolder<PluginManagerImpl> holder;
    return holder.get();
}
