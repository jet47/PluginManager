#include "gpu_module.hpp"

#include "plugin_manager.hpp"

cv::GpuModule& cv::GpuModule::instance()
{
    static Poco::SingletonHolder<cv::GpuModule> holder;

    return *holder.get();
}

Poco::SharedPtr<cv::Plugin> cv::GpuModule::getPlugin(const std::string& name)
{
    return gpuModuleManager_->getPlugin(name);
}

namespace
{
    typedef Poco::SharedPtr<cv::GpuModuleManager> (*createGpuModuleManager_t)();
}

cv::GpuModule::GpuModule()
{
    cv::PluginManager& manager = cv::PluginManager::instance();

    gpuPlugin_ = manager.getPluginSet("gpu.main").getPlugin();

    const createGpuModuleManager_t createGpuModuleManager = (createGpuModuleManager_t) gpuPlugin_->getSymbol("createGpuModuleManager");

    gpuModuleManager_ = createGpuModuleManager();
}

cv::GpuModule::~GpuModule()
{
}
