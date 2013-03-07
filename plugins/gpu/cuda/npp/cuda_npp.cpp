#include <stdexcept>

#include <npp.h>

#include "plugin.hpp"
#include "core.hpp"

#include "cuda_npp_export.h"

///////////////////////////////////////////////////////////
// Plugin Info

extern "C" CUDA_NPP_EXPORT cv::PluginInfo ocvGetPluginInfo();

cv::PluginInfo ocvGetPluginInfo()
{
    cv::PluginInfo info;

    info.name = "CUDA NPP";
    info.vendor = "Itseez";
    info.version = "2.4.4";

    info.interfaces.push_back("gpu.cuda.arithm");

    return info;
}

///////////////////////////////////////////////////////////
// gpu.cuda.arithm

extern "C" CUDA_NPP_EXPORT void gpuAddMat(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);

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

