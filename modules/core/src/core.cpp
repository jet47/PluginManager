#include "core.hpp"

#include <Poco/SingletonHolder.h>

#include "gpu_module.hpp"

cv::GpuMat::GpuMat() : rows(0), cols(0), depth(cv::d8U), channels(0), data(0), step(0)
{
}

cv::GpuMat::~GpuMat()
{
    release();
}

namespace
{
    class GpuBasicHolder : public cv::GpuBasic
    {
    public:
        GpuBasicHolder();

        void* malloc2D(size_t height, size_t width, size_t& step);
        void free(void* ptr);

    private:
        cv::Ptr<cv::GpuBasic> impl_;
    };

    GpuBasicHolder::GpuBasicHolder()
    {
        impl_ = cv::theGpuModule().create<cv::GpuBasic>("basic");
    }

    void* GpuBasicHolder::malloc2D(size_t height, size_t width, size_t& step)
    {
        impl_->malloc2D(height, width, step);
    }

    void GpuBasicHolder::free(void* ptr)
    {
        impl_->free(ptr);
    }

    GpuBasicHolder& theGpuBasic()
    {
        static Poco::SingletonHolder<GpuBasicHolder> holder;
        return *holder.get();
    }
}

void cv::GpuMat::create(int _rows, int _cols, cv::Depth _depth, int _channels)
{
    release();

    rows = _rows;
    cols = _cols;
    depth = _depth;
    channels = _channels;

    size_t type_size = depth == cv::d8U ? sizeof(unsigned char) : sizeof(float);
    data = theGpuBasic().malloc2D(rows, cols * channels * type_size, step);
}

void cv::GpuMat::release()
{
    if (data)
    {
        theGpuBasic().free(data);

        rows = 0;
        cols = 0;
        depth = cv::d8U;
        channels = 0;
        data = 0;
        step = 0;
    }
}

void cv::add(const cv::GpuMat& src1, const cv::GpuMat& src2, cv::GpuMat& dst)
{
    static cv::Ptr<cv::GpuArithmBinary> impls[2][4];

    if (src1.rows != src2.rows || src1.cols != src2.cols || src1.depth != src2.depth || src1.channels != src2.channels)
        throw std::runtime_error("Bad input");

    dst.create(src1.rows, src1.cols, src1.depth, src1.channels);

    if (impls[src1.depth][src1.channels - 1].isNull())
    {
        cv::ParameterMap params;
        params.set("func", "add_mat");
        params.set("depth", static_cast<int>(src1.depth));
        params.set("channels", src1.channels);

        impls[src1.depth][src1.channels - 1] = theGpuModule().create<cv::GpuArithmBinary>("arithm", params);
    }

    impls[src1.depth][src1.channels - 1]->apply(src1, src2, dst);
}
