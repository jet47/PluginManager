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

    class GpuMainFuncTab
    {
    public:
        static GpuMainFuncTab& instance();

        void* malloc2D(size_t height, size_t width, size_t& step);
        void free(void* ptr);

    private:
        GpuMainFuncTab();
        ~GpuMainFuncTab();

        gpuMalloc2D_t gpuMalloc2D;
        gpuFree_t gpuFree;

        friend class Poco::SingletonHolder<GpuMainFuncTab>;
    };

    GpuMainFuncTab& GpuMainFuncTab::instance()
    {
        static Poco::SingletonHolder<GpuMainFuncTab> holder;
        return *holder.get();
    }

    void* GpuMainFuncTab::malloc2D(size_t height, size_t width, size_t& step)
    {
        return gpuMalloc2D(height, width, step);
    }

    void GpuMainFuncTab::free(void* ptr)
    {
        gpuFree(ptr);
    }

    GpuMainFuncTab::GpuMainFuncTab()
    {
        cv::GpuModule& module = cv::GpuModule::instance();

        Poco::SharedPtr<cv::Plugin> plugin = module.getPlugin("main");

        gpuMalloc2D = (gpuMalloc2D_t) plugin->getSymbol("gpuMalloc2D");
        gpuFree = (gpuFree_t) plugin->getSymbol("gpuFree");
    }

    GpuMainFuncTab::~GpuMainFuncTab()
    {
    }
}

void cv::GpuMat::create(size_t rows, size_t cols, int channels)
{
    release();

    data_ = GpuMainFuncTab::instance().malloc2D(rows, cols * channels * sizeof(char), step_);
    rows_ = rows;
    cols_ = cols;
    channels_ = channels;
}

void cv::GpuMat::release()
{
    if (data_)
    {
        GpuMainFuncTab::instance().free(data_);
        data_ = 0;
        rows_ = 0;
        cols_ = 0;
        channels_ = 0;
        step_ = 0;
    }
}

namespace
{
    typedef void (*gpuAddMat_t)(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);

    class GpuArithmFuncTab
    {
    public:
        static GpuArithmFuncTab& instance();

        void add(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst);

    private:
        GpuArithmFuncTab();
        ~GpuArithmFuncTab();

        gpuAddMat_t gpuAddMat;

        friend class Poco::SingletonHolder<GpuArithmFuncTab>;
    };

    GpuArithmFuncTab& GpuArithmFuncTab::instance()
    {
        static Poco::SingletonHolder<GpuArithmFuncTab> holder;
        return *holder.get();
    }

    void GpuArithmFuncTab::add(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
    {
        gpuAddMat(src1, src2, dst);
    }

    GpuArithmFuncTab::GpuArithmFuncTab()
    {
        cv::GpuModule& module = cv::GpuModule::instance();

        Poco::SharedPtr<cv::Plugin> plugin = module.getPlugin("arithm");

        gpuAddMat = (gpuAddMat_t) plugin->getSymbol("gpuAddMat");
    }

    GpuArithmFuncTab::~GpuArithmFuncTab()
    {
    }
}

void cv::add(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
{
    GpuArithmFuncTab::instance().add(src1, src2, dst);
}
