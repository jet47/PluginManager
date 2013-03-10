#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#include "plugin_manager.hpp"
#include "core.hpp"
#include "gpu_module.hpp"

///////////////////////////////////////////////////////////
// Plugin Info

OPENCV_BEGIN_PLUGIN_DECLARATION("CUDA Main")
    OPENCV_PLUGIN_VENDOR("Itseez")
    OPENCV_PLUGIN_VERSION("2.4.4")
    OPENCV_PLUGIN_INTERFACE("gpu")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.basic")
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
// gpu

namespace
{
    class CudaModuleManager : public cv::PluginManagerBase
    {
    protected:
        cv::Ptr<cv::Object> createImpl(const std::string& interface, const cv::ParameterMap& params);
    };

    cv::Ptr<cv::Object> CudaModuleManager::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        const std::string fullName = "gpu.cuda." + interface;

        cv::PluginManager& manager = cv::thePluginManager();

        return manager.create<cv::Object>(fullName, params);
    }
}

///////////////////////////////////////////////////////////
// gpu.cuda.basic

namespace
{
    class CudaBasic : public cv::GpuBasic
    {
    public:
        void* malloc2D(size_t height, size_t width, size_t& step);
        void free(void* ptr);
    };

    void* CudaBasic::malloc2D(size_t height, size_t width, size_t& step)
    {
        void* ptr;
        cudaMallocPitch(&ptr, &step, width, height);
        return ptr;
    }

    void CudaBasic::free(void* ptr)
    {
        cudaFree(ptr);
    }
}

///////////////////////////////////////////////////////////
// ocvPluginCreate

extern "C" OPENCV_PLUGIN_API cv::Object* ocvPluginCreate(const std::string& interface, const cv::ParameterMap& params);

cv::Object* ocvPluginCreate(const std::string& interface, const cv::ParameterMap& params)
{
    assert(interface == "gpu" || interface == "gpu.cuda.basic");

    if (interface == "gpu")
        return new CudaModuleManager;

    return new CudaBasic;
}
