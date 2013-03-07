#include "core.hpp"

int main()
{
    cv::GpuMat mat;

    mat.create(100, 100, 3);
    mat.release();

    return 0;
}
