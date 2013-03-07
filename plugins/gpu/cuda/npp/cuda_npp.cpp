#include <stdexcept>

#include <npp.h>

#include "plugin_library.hpp"
#include "core.hpp"

///////////////////////////////////////////////////////////
// Plugin Info

OPENCV_BEGIN_PLUGIN_DECLARATION("CUDA NPP")
    OPENCV_PLUGIN_VENDOR("Itseez")
    OPENCV_PLUGIN_VERSION("2.4.4")
    OPENCV_PLUGIN_INTERFACE("gpu.cuda.arithm")
OPENCV_END_PLUGIN_DECLARATION()

///////////////////////////////////////////////////////////
// gpu.cuda.arithm

extern "C" OPENCV_PLUGIN_API void gpuAddMat(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);

void gpuAddMat(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
{
    if (src1.rows_ != src2.rows_ || src1.cols_ != src2.cols_ || src1.channels_ != src2.channels_)
        throw std::runtime_error("Bad input");

    dst.create(src1.rows_, src1.cols_, src1.channels_);

    NppiSize size;
    size.width = src1.cols_;
    size.height = src2.rows_;

    nppiAdd_8u_C3RSfs((const Npp8u*) src1.data_, src1.step_, (const Npp8u*) src2.data_, src2.step_, (Npp8u*) dst.data_, dst.step_, size, 0);
}

