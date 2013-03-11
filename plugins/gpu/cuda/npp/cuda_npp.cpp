#include <cassert>
#include <stdexcept>
#include <iostream>

#include <npp.h>

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
// gpu.cuda.arithm

namespace
{
    class NppAdd8UC3 : public cv::GpuArithmBinary
    {
    public:
        void apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);
    };

    void NppAdd8UC3::apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
    {
        std::cout << "NPP Add" << std::endl;

        NppiSize size;
        size.width = src1.cols;
        size.height = src2.rows;

        ::nppiAdd_8u_C3RSfs((const Npp8u*) src1.data, src1.step,
                            (const Npp8u*) src2.data, src2.step,
                            (Npp8u*) dst.data, dst.step,
                            size, 0);
    }
}

///////////////////////////////////////////////////////////
// ocvPluginCreate

extern "C" OPENCV_PLUGIN_API cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params);

cv::RefCountedObject* ocvCreatePlugin(const std::string& interface, const cv::ParameterMap& params)
{
    assert(interface == "gpu.cuda.arithm");

    const std::string func = params.get<std::string>("func");
    const cv::Depth depth = static_cast<cv::Depth>(params.get<int>("depth"));
    const int channels = params.get<int>("channels");

    if (func == "add_mat" && depth == cv::d8U && channels == 3)
        return new NppAdd8UC3;

    return 0;
}
