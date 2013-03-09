#include <cuda_runtime.h>

namespace device
{
    __global__ void kernel(const char* src1, size_t step1, const char* src2, size_t step2, char* dst, size_t dst_step, int rows, int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows)
            return;

        const float* src1_row = (const float*)(src1 + y * step1);
        const float* src2_row = (const float*)(src2 + y * step2);
        float* dst_row = (float*)(dst + y * dst_step);

        dst_row[x] = src1_row[x] + src2_row[x];
    }

    int divUp(int a, int b)
    {
        return (a + b - 1) / b;
    }

    void add(const char* src1, size_t step1, const char* src2, size_t step2, char* dst, size_t dst_step, int rows, int cols)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        kernel<<<grid, block>>>(src1, step1, src2, step2, dst, dst_step, rows, cols);

        cudaDeviceSynchronize();
    }
}
