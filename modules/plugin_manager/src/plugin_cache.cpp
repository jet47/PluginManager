#include "plugin_cache.hpp"

#include <string>
#include <iostream>
#include <fstream>

#include <Poco/File.h>
#include <Poco/Path.h>
#include <Poco/Environment.h>
#include <Poco/SharedLibrary.h>
#include <Poco/Exception.h>

#include <Poco/XML/XMLWriter.h>

#include <Poco/SAX/SAXParser.h>
#include <Poco/SAX/ContentHandler.h>
#include <Poco/SAX/Attributes.h>
#include <Poco/SAX/Locator.h>

#include "plugin_info.hpp"

namespace
{
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
}

void cv::PluginCache::update()
{
    const Poco::Path pluginBaseDirPath(Poco::Environment::get("OPENCV_PLUGIN_DIR", Poco::Path::current()));

    Poco::File pluginBaseDir(pluginBaseDirPath);

    if (pluginBaseDir.exists() && pluginBaseDir.isDirectory())
    {
        const std::string pluginManifestFile = pluginBaseDirPath.toString() + "/manifest.xml";

        std::ofstream out(pluginManifestFile.c_str());

        Poco::XML::XMLWriter writer(out, Poco::XML::XMLWriter::WRITE_XML_DECLARATION | Poco::XML::XMLWriter::PRETTY_PRINT);

        writer.startDocument();
        writer.startElement("", "", "opencv_plugins");

        ::processFolder(pluginBaseDir, writer);

        writer.endElement("", "", "opencv_plugins");
        writer.endDocument();
    }
}

namespace
{
    class ManifestParser: public Poco::XML::ContentHandler
    {
    public:
        ManifestParser(std::map<std::string, cv::PluginSet>& plugins);

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
        std::map<std::string, cv::PluginSet>& plugins_;

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

    ManifestParser::ManifestParser(std::map<std::string, cv::PluginSet>& plugins) : plugins_(plugins)
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

            Poco::SharedPtr<cv::Plugin> plugin(new cv::Plugin(info, path_));

            for (size_t i = 0; i < interfaces_.size(); ++i)
                plugins_[interfaces_[i]].add(plugin);

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
}

void cv::PluginCache::load()
{
    plugins_.clear();

    const Poco::Path pluginBaseDirPath(Poco::Environment::get("OPENCV_PLUGIN_DIR", Poco::Path::current()));

    Poco::File pluginBaseDir(pluginBaseDirPath);

    if (pluginBaseDir.exists() && pluginBaseDir.isDirectory())
    {
        const Poco::File pluginManifestFile(pluginBaseDirPath.toString() + "/manifest.xml");

        if (!pluginManifestFile.exists())
            cv::PluginCache::update();

        ManifestParser handler(plugins_);

        Poco::XML::SAXParser parser;
        parser.setContentHandler(&handler);

        try
        {
            parser.parse(pluginManifestFile.path());
        }
        catch (Poco::Exception& e)
        {
            std::cerr << e.displayText() << std::endl;
        }
    }
}
