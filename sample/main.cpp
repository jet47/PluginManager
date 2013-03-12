#include "core.hpp"
#include "plugin_manager.hpp"

int main()
{
    cv::thePluginManager()->setLogLevel(true);

    cv::GpuMat src1, src2, dst;

    src1.create(100, 100, cv::CV_8U, 3);
    src2.create(100, 100, cv::CV_8U, 3);
    cv::add(src1, src2, dst);

    src1.create(100, 100, cv::CV_32F, 1);
    src2.create(100, 100, cv::CV_32F, 1);
    cv::add(src1, src2, dst);

    return 0;
}
