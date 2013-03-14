#include <cassert>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <npp.h>

#include "utility.hpp"
#include "plugin_manager.hpp"
#include "core.hpp"
#include "gpu_module.hpp"

///////////////////////////////////////////////////////////
// Plugin Info

OPENCV_BEGIN_PLUGIN_DECLARATION("CUDA NPP")
    OPENCV_PLUGIN_VENDOR("Itseez")
    OPENCV_PLUGIN_VERSION("2.4.4")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.arithm")
OPENCV_END_PLUGIN_DECLARATION()

///////////////////////////////////////////////////////////
// NPP

namespace
{
    class Npp
    {
    public:
        static Npp* instance();

        bool load(cv::PluginLogger* logger);

        void* getSymbol(const std::string& name);

    private:
        Npp();
        ~Npp();

        cv::SharedLibrary nppLib_;

        friend class cv::SingletonHolder<Npp>;
    };

    inline Npp::Npp()
    {
    }

    inline Npp::~Npp()
    {
    }

    Npp* Npp::instance()
    {
        static cv::SingletonHolder<Npp> holder;
        return holder.get();
    }

    bool Npp::load(cv::PluginLogger* logger)
    {
        if (nppLib_.isLoaded())
            return true;

        std::ostringstream ostr;

    #if defined(OPENCV_OS_FAMILY_UNIX)
        ostr << "lib";
    #endif

        ostr << "npp";

    #if (OPENCV_OS_FAMILY_WINDOWS)
        #if (OPENCV_ARCH == OPENCV_ARCH_IA64 || OPENCV_ARCH == OPENCV_ARCH_AMD64)
            ostr << "64";
        #else
            ostr << "32";
        #endif

        ostr << "_" << NPP_VERSION_MAJOR << NPP_VERSION_MINOR;
        ostr << "_" << NPP_VERSION_BUILD;
    #endif

        ostr << cv::SharedLibrary::suffix();

    #if defined(OPENCV_OS_FAMILY_UNIX)
        ostr << "." << NPP_VERSION_MAJOR << "." << NPP_VERSION_MINOR;
    #endif

        const std::string nppLibName = ostr.str();

        ostr.str("");
        ostr << "Try to find NPP library : " << nppLibName;
        logger->message(ostr.str());

        try
        {
            nppLib_.load(nppLibName);
        }
        catch(...)
        {
            ostr.str("");
            ostr << "Can't find NPP library : " << nppLibName;
            logger->message(ostr.str());
            return false;
        }

        return true;
    }

    inline void* Npp::getSymbol(const std::string& name)
    {
        return nppLib_.getSymbol(name);
    }
}

///////////////////////////////////////////////////////////
// gpu.cuda.arithm

namespace
{
    typedef NppStatus (*nppiAdd_8u_C3_t)(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

    class NppArithmBinary : public cv::GpuArithmBinary
    {
    public:
        static bool check(const cv::ParameterMap& params, cv::PluginLogger* logger);

        NppArithmBinary();

        void apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);

    private:
        nppiAdd_8u_C3_t add_8u_c3_;
    };

    bool NppArithmBinary::check(const cv::ParameterMap& params, cv::PluginLogger* logger)
    {
        const std::string func = params.get<std::string>("func");
        const int depth = params.get<int>("depth");
        const int channels = params.get<int>("channels");

        if (func != "add_mat")
        {
            std::ostringstream msg;
            msg << "Unsopported function : " << func;
            logger->message(msg.str());
            return false;
        }

        if (depth != cv::CV_8U)
        {
            std::ostringstream msg;
            msg << "Unsopported depth : " << depth;
            logger->message(msg.str());
            return false;
        }

        if (channels != 3)
        {
            std::ostringstream msg;
            msg << "Unsopported channels count : " << channels;
            logger->message(msg.str());
            return false;
        }

        return true;
    }

    inline NppArithmBinary::NppArithmBinary()
    {
        add_8u_c3_ = (nppiAdd_8u_C3_t) Npp::instance()->getSymbol("nppiAdd_8u_C3RSfs");
    }

    void NppArithmBinary::apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
    {
        std::cout << "NPP Add" << std::endl;

        NppiSize size;
        size.width = src1.cols;
        size.height = src2.rows;

        add_8u_c3_((const Npp8u*) src1.data, (int) src1.step,
                   (const Npp8u*) src2.data, (int) src2.step,
                   (Npp8u*) dst.data, (int) dst.step,
                    size, 0);
    }
}

///////////////////////////////////////////////////////////
// ocvPluginCreate

extern "C" OPENCV_PLUGIN_API cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger);

cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params, cv::PluginLogger* logger)
{
    assert(interface == "gpu.cuda.arithm");

    // check that we have NPP
    if (!Npp::instance()->load(logger))
        return 0;

    if (!NppArithmBinary::check(params, logger))
        return 0;

    return new NppArithmBinary;
}
