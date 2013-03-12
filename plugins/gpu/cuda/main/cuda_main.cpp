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
    OPENCV_PLUGIN_INTERFACE("gpu.module")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.basic")
OPENCV_END_PLUGIN_DECLARATION()

///////////////////////////////////////////////////////////
// gpu

namespace
{
    class CudaModuleManager : public cv::PluginManagerBase
    {
    public:
        CudaModuleManager();

    protected:
        cv::AutoPtr<cv::RefCountedObject> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        cv::PluginManager* manager_;
    };

    CudaModuleManager::CudaModuleManager()
    {
        manager_ = cv::thePluginManager();
    }

    cv::AutoPtr<cv::RefCountedObject> CudaModuleManager::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        const std::string fullName = "gpu.cuda." + interface;

        return manager_->create<cv::RefCountedObject>(fullName, params);
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

extern "C" OPENCV_PLUGIN_API cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger);

cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* /*logger*/)
{
    assert(interface == "gpu.module" || interface == "gpu.cuda.basic");

    if (interface == "gpu.module")
        return new CudaModuleManager;

    return new CudaBasic;
}
