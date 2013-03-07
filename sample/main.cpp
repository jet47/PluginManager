#include "core.hpp"

int main()
{
    cv::GpuMat src1, src2, dst;

    src1.create(100, 100, 3);
    src2.create(100, 100, 3);

    cv::add(src1, src2, dst);

    return 0;
}
