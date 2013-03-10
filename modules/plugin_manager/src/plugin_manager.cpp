#include "plugin_manager.hpp"

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <utility>

#include <Poco/SingletonHolder.h>
#include <Poco/Environment.h>
#include <Poco/Path.h>
#include <Poco/File.h>
#include <Poco/SharedLibrary.h>

#include <Poco/XML/XMLWriter.h>

#include <Poco/SAX/SAXParser.h>
#include <Poco/SAX/ContentHandler.h>
#include <Poco/SAX/Attributes.h>
#include <Poco/SAX/Locator.h>

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
        void updateCache(const std::string& baseDir, const std::string& manifestFile);
        void loadPluginCache(const std::string& baseDir, const std::string& manifestFile);

        void getPluginList(std::vector<cv::Ptr<cv::PluginBase> >& plugins);

        void setPriority(const std::string& interface, const std::string& pluginName, int priority);

    protected:
        cv::Ptr<cv::Object> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        std::vector<cv::Ptr<cv::Plugin> > allPlugins_;
        std::map<std::string, std::vector<cv::Ptr<cv::Plugin> > > pluginsMap_;
        std::map<std::string, std::vector<std::pair<std::string, int> > > allPriorityMap_;
    };

    Poco::File parseBaseDir(const std::string& baseDir)
    {
        if (!baseDir.empty())
            return baseDir;

        return Poco::Environment::get("OPENCV_PLUGIN_DIR", Poco::Path::current());
    }

    std::string parseManifestFile(const std::string& pluginBaseDir, const std::string& manifestFile)
    {
        if (manifestFile.empty())
            return pluginBaseDir + Poco::Path::separator() + "manifest.xml";

        Poco::Path path(manifestFile);
        if (path.isAbsolute())
            return manifestFile;

        return pluginBaseDir + Poco::Path::separator() + manifestFile;
    }

    typedef cv::PluginInfo (*ocvGetPluginInfo_t)();

    void processFolder(const Poco::File& folder, Poco::XML::XMLWriter& writer)
    {
        std::vector<Poco::File> files;
        folder.list(files);

        for (size_t i = 0; i < files.size(); ++i)
        {
            const Poco::File& file = files[i];

            if (file.isDirectory())
                processFolder(file, writer);
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

                    writer.startElement("", "", "plugin");

                    writer.startElement("", "", "path");
                    writer.characters(path);
                    writer.endElement("", "", "path");

                    writer.startElement("", "", "name");
                    writer.characters(info.name);
                    writer.endElement("", "", "name");

                    writer.startElement("", "", "vendor");
                    writer.characters(info.vendor);
                    writer.endElement("", "", "vendor");

                    writer.startElement("", "", "version");
                    writer.characters(info.version);
                    writer.endElement("", "", "version");

                    writer.startElement("", "", "interfaces");
                    for (size_t i = 0; i < info.interfaces.size(); ++i)
                    {
                        writer.startElement("", "", "interface");
                        writer.characters(info.interfaces[i]);
                        writer.endElement("", "", "interface");
                    }
                    writer.endElement("", "", "interfaces");

                    writer.endElement("", "", "plugin");
                }
                catch(const Poco::Exception& e)
                {
                    std::cerr << e.message() << std::endl;
                }
            }
        }
    }

    void PluginManagerImpl::updateCache(const std::string& baseDir, const std::string& manifestFile)
    {
        const Poco::File pluginBaseDir = ::parseBaseDir(baseDir);

        if (pluginBaseDir.exists() && pluginBaseDir.isDirectory())
        {
            const std::string pluginManifestFile = ::parseManifestFile(pluginBaseDir.path(), manifestFile);

            std::ofstream out(pluginManifestFile.c_str());

            Poco::XML::XMLWriter writer(out, Poco::XML::XMLWriter::WRITE_XML_DECLARATION | Poco::XML::XMLWriter::PRETTY_PRINT);

            writer.startDocument();
            writer.startElement("", "", "opencv_plugins");

            ::processFolder(pluginBaseDir, writer);

            writer.endElement("", "", "opencv_plugins");
            writer.endDocument();
        }
    }

    class ManifestParser: public Poco::XML::ContentHandler
    {
    public:
        ManifestParser(std::vector<cv::Ptr<cv::Plugin> >& plugins);

        void setDocumentLocator(const Poco::XML::Locator* loc);

        void startDocument();

        void endDocument();

        void startElement(const Poco::XML::XMLString& uri, const Poco::XML::XMLString& localName, const Poco::XML::XMLString& qname, const Poco::XML::Attributes& attributes);

        void endElement(const Poco::XML::XMLString& uri, const Poco::XML::XMLString& localName, const Poco::XML::XMLString& qname);

        void characters(const Poco::XML::XMLChar ch[], int start, int length);

        void ignorableWhitespace(const Poco::XML::XMLChar ch[], int start, int length);

        void processingInstruction(const Poco::XML::XMLString& target, const Poco::XML::XMLString& data);

        void startPrefixMapping(const Poco::XML::XMLString& prefix, const Poco::XML::XMLString& uri);

        void endPrefixMapping(const Poco::XML::XMLString& prefix);

        void skippedEntity(const Poco::XML::XMLString& name);

    private:
        std::vector<cv::Ptr<cv::Plugin> >& plugins_;

        bool in_opencv_plugins_;
        bool in_plugin_;
        bool in_path_;
        bool in_name_;
        bool in_vendor_;
        bool in_version_;
        bool in_interfaces_;
        bool in_interface_;

        std::string path_;
        std::string name_;
        std::string vendor_;
        std::string version_;
        std::vector<std::string> interfaces_;

        std::string interface_;
    };

    ManifestParser::ManifestParser(std::vector<cv::Ptr<cv::Plugin> >& plugins) : plugins_(plugins)
    {
        in_opencv_plugins_ = false;
        in_plugin_ = false;
        in_path_ = false;
        in_name_ = false;
        in_vendor_ = false;
        in_version_ = false;
        in_interfaces_ = false;
        in_interface_ = false;
    }

    void ManifestParser::setDocumentLocator(const Poco::XML::Locator* loc)
    {
    }

    void ManifestParser::startDocument()
    {
    }

    void ManifestParser::endDocument()
    {
    }

    void ManifestParser::startElement(const Poco::XML::XMLString& uri, const Poco::XML::XMLString& localName, const Poco::XML::XMLString& qname, const Poco::XML::Attributes& attributes)
    {
        if (localName == "opencv_plugins")
        {
            in_opencv_plugins_ = true;

            plugins_.clear();
        }
        else if (localName == "plugin")
        {
            if (!in_opencv_plugins_)
                throw std::runtime_error("Incorrect manifest file");

            in_plugin_ = true;
        }
        else if (localName == "path")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            in_path_ = true;

            path_ = "";
        }
        else if (localName == "name")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            in_name_ = true;

            name_ = "";
        }
        else if (localName == "vendor")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            in_vendor_ = true;

            vendor_ = "";
        }
        else if (localName == "version")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            in_version_ = true;

            version_ = "";
        }
        else if (localName == "interfaces")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            in_interfaces_ = true;

            interfaces_.clear();
        }
        else if (localName == "interface")
        {
            if (!in_interfaces_)
                throw std::runtime_error("Incorrect manifest file");

            in_interface_ = true;

            interface_ = "";
        }
    }

    void ManifestParser::endElement(const Poco::XML::XMLString& uri, const Poco::XML::XMLString& localName, const Poco::XML::XMLString& qname)
    {
        if (localName == "opencv_plugins")
        {
            if (!in_opencv_plugins_)
                throw std::runtime_error("Incorrect manifest file");

            in_opencv_plugins_ = false;
        }
        else if (localName == "plugin")
        {
            if (!in_plugin_)
                throw std::runtime_error("Incorrect manifest file");

            cv::PluginInfo info;

            info.name = name_;
            info.vendor = vendor_;
            info.version = version_;
            info.interfaces = interfaces_;

            cv::Ptr<cv::Plugin> plugin = new cv::Plugin(info, path_);
            plugins_.push_back(plugin);

            in_plugin_ = false;
        }
        else if (localName == "path")
        {
            if (!in_path_)
                throw std::runtime_error("Incorrect manifest file");

            in_path_ = false;
        }
        else if (localName == "name")
        {
            if (!in_name_)
                throw std::runtime_error("Incorrect manifest file");

            in_name_ = false;
        }
        else if (localName == "vendor")
        {
            if (!in_vendor_)
                throw std::runtime_error("Incorrect manifest file");

            in_vendor_ = false;
        }
        else if (localName == "version")
        {
            if (!in_version_)
                throw std::runtime_error("Incorrect manifest file");

            in_version_ = false;
        }
        else if (localName == "interfaces")
        {
            if (!in_interfaces_)
                throw std::runtime_error("Incorrect manifest file");

            in_interfaces_ = false;
        }
        else if (localName == "interface")
        {
            if (!in_interface_)
                throw std::runtime_error("Incorrect manifest file");

            interfaces_.push_back(interface_);

            in_interface_ = false;
        }
    }

    void ManifestParser::characters(const Poco::XML::XMLChar ch[], int start, int length)
    {
        if (in_path_)
            path_ += std::string(ch + start, length);
        else if (in_name_)
            name_ += std::string(ch + start, length);
        else if (in_vendor_)
            vendor_ += std::string(ch + start, length);
        else if (in_version_)
            version_ += std::string(ch + start, length);
        else if (in_interface_)
            interface_ += std::string(ch + start, length);
    }

    void ManifestParser::ignorableWhitespace(const Poco::XML::XMLChar[], int, int)
    {
    }

    void ManifestParser::processingInstruction(const Poco::XML::XMLString&, const Poco::XML::XMLString&)
    {
    }

    void ManifestParser::startPrefixMapping(const Poco::XML::XMLString&, const Poco::XML::XMLString&)
    {
    }

    void ManifestParser::endPrefixMapping(const Poco::XML::XMLString&)
    {
    }

    void ManifestParser::skippedEntity(const Poco::XML::XMLString&)
    {
    }

    void PluginManagerImpl::loadPluginCache(const std::string& baseDir, const std::string& manifestFile)
    {
        const Poco::File pluginBaseDir = ::parseBaseDir(baseDir);

        if (pluginBaseDir.exists() && pluginBaseDir.isDirectory())
        {
            const Poco::File pluginManifestFile = ::parseManifestFile(pluginBaseDir.path(), manifestFile);

            if (!pluginManifestFile.exists())
                updateCache(baseDir, manifestFile);

            ManifestParser handler(allPlugins_);

            Poco::XML::SAXParser parser;
            parser.setContentHandler(&handler);

            try
            {
                parser.parse(pluginManifestFile.path());

                pluginsMap_.clear();
                for (size_t i = 0; i < allPlugins_.size(); ++i)
                {
                    cv::Ptr<cv::Plugin> plugin = allPlugins_[i];
                    cv::PluginInfo info = plugin->info();

                    for (size_t j = 0; j < info.interfaces.size(); ++j)
                        pluginsMap_[info.interfaces[j]].push_back(plugin);
                }
            }
            catch (Poco::Exception& e)
            {
                std::cerr << e.displayText() << std::endl;
            }
        }
    }

    void PluginManagerImpl::getPluginList(std::vector<cv::Ptr<cv::PluginBase> >& plugins)
    {
        plugins.clear();

        std::copy(allPlugins_.begin(), allPlugins_.end(), std::back_inserter(plugins));
    }

    void PluginManagerImpl::setPriority(const std::string& interface, const std::string& pluginName, int priority)
    {
        std::vector<std::pair<std::string, int> >& vec = allPriorityMap_[interface];

        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (vec[i].first == pluginName)
            {
                vec[i].second = priority;
                return;
            }
        }

        vec.push_back(std::make_pair(pluginName, priority));
    }

    struct PluginCompare
    {
        mutable std::map<std::string, int> priorityMap;

        bool operator ()(const cv::Ptr<cv::Plugin>& pluginA, const cv::Ptr<cv::Plugin>& pluginB) const
        {
            return priorityMap[pluginA->info().name] > priorityMap[pluginB->info().name];
        }
    };

    cv::Ptr<cv::Object> PluginManagerImpl::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        if (allPlugins_.empty())
            loadPluginCache("", "");

        std::vector<cv::Ptr<cv::Plugin> > plugins = pluginsMap_[interface];

        if (plugins.empty())
        {
            std::ostringstream msg;
            msg << "Can't find plugin for interface : " << interface;
            throw std::runtime_error(msg.str());
        }

        PluginCompare pluginCmp;
        for (size_t i = 0; i < plugins.size(); ++i)
        {
            const cv::Ptr<cv::Plugin>& plugin = plugins[i];
            pluginCmp.priorityMap[plugin->info().name] = plugin->isLoaded();
        }
        std::vector<std::pair<std::string, int> >& userPriority = allPriorityMap_[interface];
        for (size_t i = 0; i < userPriority.size(); ++i)
        {
            const std::pair<std::string, int>& pair = userPriority[i];
            pluginCmp.priorityMap[pair.first] = pair.second;
        }

        std::sort(plugins.begin(), plugins.end(), pluginCmp);

        for (size_t i = 0; i < plugins.size(); ++i)
        {
            cv::Ptr<cv::Plugin>& plugin = plugins[i];

            if (pluginCmp.priorityMap[plugin->info().name] < 0)
                break;

            bool wasLoaded = plugin->isLoaded();

            if (!wasLoaded && !plugin->load())
                continue;

            cv::Ptr<cv::Object> obj = plugin->create(interface, params);
            if (!obj.isNull())
                return obj;

            if (!wasLoaded)
                plugin->unload();
        }

        std::ostringstream msg;
        msg << "Can't find plugin for interface : " << interface;
        throw std::runtime_error(msg.str());
    }
}

cv::PluginManager& cv::thePluginManager()
{
    static Poco::SingletonHolder<PluginManagerImpl> holder;
    return *holder.get();
}
