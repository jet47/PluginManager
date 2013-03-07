#include "plugin_cache.hpp"

#include <string>
#include <iostream>
#include <fstream>

#include <Poco/File.h>
#include <Poco/Path.h>
#include <Poco/Environment.h>
#include <Poco/SharedLibrary.h>

#include <Poco/XML/XMLWriter.h>

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

                    writer.startElement("", "", "interfraces");
                    for (size_t i = 0; i < info.interfaces.size(); ++i)
                    {
                        writer.startElement("", "", "interfrace");
                        writer.characters(info.interfaces[i]);
                        writer.endElement("", "", "interfrace");
                    }
                    writer.endElement("", "", "interfraces");

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
    }
}
