#include <cuda_runtime.h>

#include "plugin.hpp"
#include "plugin_manager.hpp"
#include "gpu_module.hpp"

#include "cuda_main_export.h"

///////////////////////////////////////////////////////////
// Plugin Info

extern "C" CUDA_MAIN_EXPORT cv::PluginInfo ocvGetPluginInfo();

cv::PluginInfo ocvGetPluginInfo()
{
    cv::PluginInfo info;

    info.name = "CUDA Main";
    info.vendor = "Itseez";
    info.version = "2.4.4";

    info.interfaces.push_back("gpu.main");
    info.interfaces.push_back("gpu.cuda.main");

    return info;
}

///////////////////////////////////////////////////////////
// gpu.main

extern "C" CUDA_MAIN_EXPORT Poco::SharedPtr<cv::GpuModuleManager> createGpuModuleManager();

namespace
{
    class CUDA_MAIN_EXPORT CudaModuleManager : public cv::GpuModuleManager
    {
    public:
        Poco::SharedPtr<cv::Plugin> getPlugin(const std::string& name);
    };

    Poco::SharedPtr<cv::Plugin> CudaModuleManager::getPlugin(const std::string& name)
    {
        const std::string fullName = "gpu.cuda." + name;

        cv::PluginManager& manager = cv::PluginManager::instance();

        cv::PluginSet& set = manager.getPluginSet(fullName);

        return set.getPlugin();
    }
}

Poco::SharedPtr<cv::GpuModuleManager> createGpuModuleManager()
{
    return new CudaModuleManager;
}

///////////////////////////////////////////////////////////
// gpu.cuda.main

extern "C" CUDA_MAIN_EXPORT void* gpuMalloc2D(size_t height, size_t width, size_t& step);
extern "C" CUDA_MAIN_EXPORT void gpuFree(void* ptr);

void* gpuMalloc2D(size_t height, size_t width, size_t& step)
{
    void* ptr;
    cudaMallocPitch(&ptr, &step, width, height);
    return ptr;
}

void gpuFree(void* ptr)
{
    cudaFree(ptr);
}
