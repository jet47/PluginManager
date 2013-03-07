#include <iostream>

#include <cuda_runtime.h>

#include "plugin_library.hpp"
#include "plugin_manager.hpp"
#include "gpu_module.hpp"

///////////////////////////////////////////////////////////
// Plugin Info

OPENCV_BEGIN_PLUGIN_DECLARATION("CUDA Main")
    OPENCV_PLUGIN_VENDOR("Itseez")
    OPENCV_PLUGIN_VERSION("2.4.4")
    OPENCV_PLUGIN_INTERFACE("gpu.main")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.main")
OPENCV_END_PLUGIN_DECLARATION()

bool ocvLoadPlugin()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
    {
        std::cerr << "CUDA : Insufficient Driver" << std::endl;
        return false;
    }

    if (error == cudaErrorNoDevice)
    {
        std::cerr << "CUDA : No Device" << std::endl;
        return false;
    }

    return (count > 0);
}

///////////////////////////////////////////////////////////
// gpu.main

extern "C" OPENCV_PLUGIN_API Poco::SharedPtr<cv::GpuModuleManager> createGpuModuleManager();

namespace
{
    class CudaModuleManager : public cv::GpuModuleManager
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

extern "C" OPENCV_PLUGIN_API void* gpuMalloc2D(size_t height, size_t width, size_t& step);
extern "C" OPENCV_PLUGIN_API void gpuFree(void* ptr);

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
