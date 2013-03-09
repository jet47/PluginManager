#include <cassert>
#include <stdexcept>

#include "plugin_manager.hpp"
#include "core.hpp"
#include "gpu_module.hpp"

///////////////////////////////////////////////////////////
// Plugin Info

OPENCV_BEGIN_PLUGIN_DECLARATION("CUDA Arithm")
    OPENCV_PLUGIN_VENDOR("Itseez")
    OPENCV_PLUGIN_VERSION("2.4.4")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.arithm")
OPENCV_END_PLUGIN_DECLARATION()

///////////////////////////////////////////////////////////
// gpu.cuda.arithm

namespace device
{
    void add(const char* src1, size_t step1, const char* src2, size_t step2, char* dst, size_t dst_step, int rows, int cols);
}

namespace
{
    class Add32FC1 : public cv::GpuArithmBinary
    {
    public:
        void apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);
    };

    void Add32FC1::apply(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
    {
        device::add((const char*) src1.data, src1.step, (const char*) src2.data, src2.step, (char*) dst.data, dst.step, src1.rows, src1.cols);
    }
}

///////////////////////////////////////////////////////////
// ocvPluginCreate

extern "C" OPENCV_PLUGIN_API cv::Ptr<cv::Object> ocvPluginCreate(const std::string& interface, const cv::ParameterMap& params);

cv::Ptr<cv::Object> ocvPluginCreate(const std::string& interface, const cv::ParameterMap& params)
{
    assert(interface == "gpu.cuda.arithm");

    const std::string func = params.get<std::string>("func");
    const cv::Depth depth = static_cast<cv::Depth>(params.get<int>("depth"));
    const int channels = params.get<int>("channels");

    if (func == "add_mat" && depth == cv::d32F && channels == 1)
        return new Add32FC1;

    return cv::Ptr<cv::Object>();
}
