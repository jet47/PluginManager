#include <iostream>
#include <vector>

#include <Poco/Util/Application.h>
#include <Poco/Util/HelpFormatter.h>

#include "plugin_manager.hpp"

namespace
{
    class PluginManagerApp : public Poco::Util::Application
    {
    public:
        PluginManagerApp();

    protected:
        void defineOptions(Poco::Util::OptionSet& options);
        int main(const std::vector<std::string>& args);

    private:
        void printHelp();

        void handleHelp(const std::string& name, const std::string& value);
        void handleUpdate(const std::string& name, const std::string& value);
        void handleInfo(const std::string& name, const std::string& value);

        bool doneSmt_;
    };

    PluginManagerApp::PluginManagerApp() : doneSmt_(false)
    {
    }

    void PluginManagerApp::defineOptions(Poco::Util::OptionSet& options)
    {
        Poco::Util::Application::defineOptions(options);

        options.addOption(Poco::Util::Option()
                          .fullName("help")
                          .shortName("h")
                          .description("display help information")
                          .required(false)
                          .repeatable(false)
                          .noArgument()
                          .callback(Poco::Util::OptionCallback<PluginManagerApp>(this, &PluginManagerApp::handleHelp)));

        options.addOption(Poco::Util::Option()
                          .fullName("update")
                          .description("update plugin cache")
                          .required(false)
                          .repeatable(false)
                          .argument("plugin directory", false)
                          .callback(Poco::Util::OptionCallback<PluginManagerApp>(this, &PluginManagerApp::handleUpdate)));

        options.addOption(Poco::Util::Option()
                          .fullName("info")
                          .description("print info about all plugins")
                          .required(false)
                          .repeatable(false)
                          .argument("manifest file", false)
                          .callback(Poco::Util::OptionCallback<PluginManagerApp>(this, &PluginManagerApp::handleInfo)));
    }

    void PluginManagerApp::printHelp()
    {
        Poco::Util::HelpFormatter helpFormatter(options());
        helpFormatter.setCommand(commandName());
        helpFormatter.setUsage("OPTIONS");
        helpFormatter.setHeader("OpenCV Plugin Manager");
        helpFormatter.format(std::cout);
    }

    void PluginManagerApp::handleHelp(const std::string& name, const std::string& value)
    {
        doneSmt_ = true;

        printHelp();
        stopOptionsProcessing();
    }

    void PluginManagerApp::handleUpdate(const std::string& name, const std::string& value)
    {
        doneSmt_ = true;

        cv::thePluginManager().updateCache(value);
    }

    void PluginManagerApp::handleInfo(const std::string& name, const std::string& value)
    {
        doneSmt_ = true;

        cv::thePluginManager().loadPluginCache(value);

        std::vector<cv::Ptr<cv::PluginBase> > plugins;
        cv::thePluginManager().getPluginList(plugins);

        std::cout << "OpenCV Plugins: \n" << std::endl;
        for (size_t i = 0; i < plugins.size(); ++i)
        {
            cv::Ptr<cv::PluginBase> plugin = plugins[i];
            cv::PluginInfo info = plugin->info();

            std::cout << "\t " << info.name << std::endl;
            std::cout << "\t\t Version : " << info.version << std::endl;
            std::cout << "\t\t Vendor : " << info.vendor << std::endl;
            std::cout << "\t\t Interfaces : " << std::endl;

            for (size_t j = 0; j < info.interfaces.size(); ++j)
                std::cout << "\t\t\t " << info.interfaces[j] << std::endl;

            std::cout << std::endl;
        }
    }

    int PluginManagerApp::main(const std::vector<std::string>& args)
    {
        if (!doneSmt_)
            printHelp();

        return EXIT_OK;
    }
}

POCO_APP_MAIN(PluginManagerApp)
