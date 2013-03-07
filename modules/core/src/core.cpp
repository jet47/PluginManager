#include "core.hpp"

#include <Poco/SingletonHolder.h>

#include "gpu_module.hpp"

cv::GpuMat::GpuMat() : rows_(0), cols_(0), channels_(0), data_(0), step_(0)
{
}

cv::GpuMat::~GpuMat()
{
    release();
}

namespace
{
    typedef void* (*gpuMalloc2D_t)(size_t height, size_t width, size_t& step);
    typedef void (*gpuFree_t)(void* ptr);

    class GpuFuncTab
    {
    public:
        static GpuFuncTab& instance();

        void* malloc2D(size_t height, size_t width, size_t& step);
        void free(void* ptr);

    private:
        GpuFuncTab();
        ~GpuFuncTab();

        gpuMalloc2D_t gpuMalloc2D;
        gpuFree_t gpuFree;

        friend class Poco::SingletonHolder<GpuFuncTab>;
    };

    GpuFuncTab& GpuFuncTab::instance()
    {
        static Poco::SingletonHolder<GpuFuncTab> holder;
        return *holder.get();
    }

    void* GpuFuncTab::malloc2D(size_t height, size_t width, size_t& step)
    {
        return gpuMalloc2D(height, width, step);
    }

    void GpuFuncTab::free(void* ptr)
    {
        gpuFree(ptr);
    }

    GpuFuncTab::GpuFuncTab()
    {
        cv::GpuModule& module = cv::GpuModule::instance();

        Poco::SharedPtr<cv::Plugin> plugin = module.getPlugin("main");

        gpuMalloc2D = (gpuMalloc2D_t) plugin->getSymbol("gpuMalloc2D");
        gpuFree = (gpuFree_t) plugin->getSymbol("gpuFree");
    }

    GpuFuncTab::~GpuFuncTab()
    {
    }
}

void cv::GpuMat::create(size_t rows, size_t cols, int channels)
{
    release();

    data_ = GpuFuncTab::instance().malloc2D(rows, cols * channels * sizeof(char), step_);
    rows_ = rows;
    cols_ = cols;
    channels_ = channels;
}

void cv::GpuMat::release()
{
    if (data_)
    {
        GpuFuncTab::instance().free(data_);
        data_ = 0;
        rows_ = 0;
        cols_ = 0;
        channels_ = 0;
        step_ = 0;
    }
}
