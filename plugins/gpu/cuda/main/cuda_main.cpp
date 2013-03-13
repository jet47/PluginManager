#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

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
// Cuda

namespace
{
    typedef cudaError_t (CUDARTAPI *cudaGetDeviceCount_t)(int *count);

    class Cuda
    {
    public:
        static Cuda* instance();

        bool load(cv::PluginLogger* logger);
        bool checkDevice(cv::PluginLogger* logger);

        void* getSymbol(const std::string& name);

    private:
        Cuda();
        ~Cuda();

        cv::SharedLibrary cudaLib_;
        cudaGetDeviceCount_t getDeviceCount_;

        friend class cv::SingletonHolder<Cuda>;
    };

    inline Cuda::Cuda()
    {
    }

    inline Cuda::~Cuda()
    {
    }

    Cuda* Cuda::instance()
    {
        static cv::SingletonHolder<Cuda> holder;
        return holder.get();
    }

    bool Cuda::load(cv::PluginLogger* logger)
    {
        if (cudaLib_.isLoaded())
            return true;

        std::ostringstream ostr;

    #if defined(OPENCV_OS_FAMILY_UNIX)
        ostr << "lib";
    #endif

        ostr << "cudart";

    #if (OPENCV_ARCH == OPENCV_ARCH_IA64 || OPENCV_ARCH == OPENCV_ARCH_AMD64)
        ostr << "64";
    #else
        ostr << "32";
    #endif

        ostr << "_" << (CUDART_VERSION / 100);
        ostr << "_" << NPP_VERSION_BUILD;
        ostr << cv::SharedLibrary::suffix();

        const std::string cudaLibName = ostr.str();

        ostr.str("");
        ostr << "Try to find CUDA library : " << cudaLibName;
        logger->message(ostr.str());

        try
        {
            cudaLib_.load(cudaLibName);
            getDeviceCount_ = (cudaGetDeviceCount_t) cudaLib_.getSymbol("cudaGetDeviceCount");
        }
        catch(const std::exception& e)
        {
            ostr.str("");
            ostr << "Can't find CUDA library : " << cudaLibName;
            logger->message(ostr.str());
            logger->message(e.what());
            return false;
        }

        return true;
    }

    inline void* Cuda::getSymbol(const std::string& name)
    {
        return cudaLib_.getSymbol(name);
    }

    bool Cuda::checkDevice(cv::PluginLogger* logger)
    {
        if (!load(logger))
            return false;

        int count;
        cudaError_t error = getDeviceCount_( &count );

        if (error == cudaErrorInsufficientDriver)
        {
            logger->message("Insufficient driver");
            return false;
        }

        if (error == cudaErrorNoDevice)
        {
            logger->message("No CUDA device");
            return false;
        }

        return count > 0;
    }
}

///////////////////////////////////////////////////////////
// gpu.module

namespace
{
    class CudaModuleManager : public cv::PluginManagerBase
    {
    public:
        CudaModuleManager();

        void init();

    protected:
        cv::AutoPtr<cv::RefCountedObject> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        cv::PluginManager* manager_;
    };

    CudaModuleManager::CudaModuleManager()
    {
        manager_ = cv::thePluginManager();
    }

    void CudaModuleManager::init()
    {
        std::string pluginDir = cv::Environment::get("OPENCV_PLUGIN_DIR", ".");

        pluginDir += "/gpu/cuda";

        #if defined(OPENCV_OS_FAMILY_WINDOWS)
            #ifdef _DEBUG
                pluginDir += "\\Debug";
            #else
                pluginDir += "\\Release";
            #endif
        #endif

        manager_->addPluginDir(pluginDir, false);
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
    typedef cudaError_t (CUDARTAPI *cudaMallocPitch_t)(void **devPtr, size_t *pitch, size_t width, size_t height);
    typedef cudaError_t (CUDARTAPI *cudaFree_t)(void *devPtr);

    class CudaBasic : public cv::GpuBasic
    {
    public:
        CudaBasic();

        void* malloc2D(size_t height, size_t width, size_t& step);
        void free(void* ptr);

    private:
        cudaMallocPitch_t malloc2D_;
        cudaFree_t free_;
    };

    inline CudaBasic::CudaBasic()
    {
        malloc2D_ = (cudaMallocPitch_t) Cuda::instance()->getSymbol("cudaMallocPitch");
        free_ = (cudaFree_t) Cuda::instance()->getSymbol("cudaFree");
    }

    void* CudaBasic::malloc2D(size_t height, size_t width, size_t& step)
    {
        void* ptr;
        malloc2D_(&ptr, &step, width, height);
        return ptr;
    }

    void CudaBasic::free(void* ptr)
    {
        free_(ptr);
    }
}

///////////////////////////////////////////////////////////
// ocvPluginCreate

extern "C" OPENCV_PLUGIN_API cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger);

cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger)
{
    assert(interface == "gpu.module" || interface == "gpu.cuda.basic");

    if (interface == "gpu.module")
    {
        // check that we have CUDA
        if (!Cuda::instance()->load(logger))
            return 0;

        // check that we have CUDA device
        if (!Cuda::instance()->checkDevice(logger))
            return 0;

        return new CudaModuleManager;
    }

    return new CudaBasic;
}
