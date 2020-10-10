
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <type_traits>

namespace {

template <typename scalar_t>
__device__ scalar_t diff_of_products (scalar_t a, scalar_t b, scalar_t c, scalar_t d,
                                      typename std::enable_if<std::is_same<scalar_t, float>::value>::type * = 0)
{
    scalar_t w = d * c;
    float e = fmaf(-d, c, w);
    float f = fmaf(a, b, -w);
    return f + e;
}

template <typename scalar_t>
__device__ scalar_t diff_of_products (scalar_t a, scalar_t b, scalar_t c, scalar_t d,
                                      typename std::enable_if<std::is_same<scalar_t, double>::value>::type * = 0)
{
    scalar_t w = d * c;
    double e = fma(-d, c, w);
    double f = fma(a, b, -w);
    return f + e;
}

template <typename scalar_t>
__device__ scalar_t sum_of_products (scalar_t a, scalar_t b, scalar_t c, scalar_t d,
                                     typename std::enable_if<std::is_same<scalar_t, float>::value>::type * = 0)
{
    scalar_t w = d * c;
    float e = fmaf(d, c, w);
    float f = fmaf(a, b, -w);
    return f + e;
}

template <typename scalar_t>
__device__ scalar_t sum_of_products (scalar_t a, scalar_t b, scalar_t c, scalar_t d,
                                     typename std::enable_if<std::is_same<scalar_t, double>::value>::type * = 0)
{
    scalar_t w = d * c;
    double e = fma(d, c, w);
    double f = fma(a, b, -w);
    return f + e;
}

template<typename scalar_t>
__device__ float intersection_area(const scalar_t* rect1, const scalar_t* rect2) {
    float intersection[50 * 2] = {0};
    float line_v[50] = {0};
    float new_intersec[50 * 2] = {0};

    // I use loop based method, it is not elegant
    // TODO need to fix
//    cudaMemcpy(&intersection, rect1, sizeof(float) * 8, cudaMemcpyDeviceToDevice);

    for (int i = 0; i < 4; i++) {
        intersection[2 * i + 0] = rect1[2 * i + 0];
        intersection[2 * i + 1] = rect1[2 * i + 1];
    }
    int cur_intersection_size = 4;
    // TODO use shared memory??
    for (int i = 0; i < 4; i++) {
        if (cur_intersection_size < 2)
            break;
        float x1 = rect2[i * 2 + 0];
        float y1 = rect2[i * 2 + 1];

        float x2 = rect2[(i + 1) % 4 * 2 + 0];
        float y2 = rect2[(i + 1) % 4 * 2 + 1];

        float a = y2 - y1;
        float b = x1 - x2;

        float c = diff_of_products<float>(x2, y1, y2, x1);


        for (int j = 0; j < cur_intersection_size; ++j) {
            float c_x = intersection[j * 2 + 0];
            float c_y = intersection[j * 2 + 1];
            line_v[j] = sum_of_products<float>(a, c_x, b, c_y) + c;
        }

        int new_size = 0;
        for (int j = 0; j < cur_intersection_size; ++j) {
            float s_value = line_v[j];
            float t_value = line_v[(j + 1) % (cur_intersection_size)];
            if (s_value <= 0) {
                new_intersec[2 * new_size + 0] = intersection[2 * j + 0];
                new_intersec[2 * new_size + 1] = intersection[2 * j + 1];
                new_size += 1;
            }
            if (s_value * t_value < 0) {
                float x3 = intersection[j * 2 + 0];
                float y3 = intersection[j * 2 + 1];

                float x4 = intersection[(j + 1) % (cur_intersection_size) * 2 + 0];
                float y4 = intersection[(j + 1) % (cur_intersection_size) * 2 + 1];
                float a1 = y4 - y3;
                float b1 = x3 - x4;
                float c1 = diff_of_products<float>(x4, y3, y4, x3);
                float w = diff_of_products<float>(a, b1, b, a1);

                if (w == 0) {
                    //printf("hit w = 0 %f, %f, %f, %f\n", a, b1, b, a1);
                    continue;
                }
                float c_x = diff_of_products<float>(b, c1, c, b1) / w;
                float c_y = diff_of_products<float>(c, a1, a, c1) / w;
                new_intersec[2 * new_size + 0] = c_x;
                new_intersec[2 * new_size + 1] = c_y;
                new_size += 1;

            }
        }
        for (int k = 0; k < new_size; k++) {
            intersection[2 * k + 0] = new_intersec[2 * k + 0];
            intersection[2 * k + 1] = new_intersec[2 * k + 1];
        }
//        cudaMemcpy(&intersection, &new_intersec, sizeof(float) * new_size * 2, cudaMemcpyDeviceToDevice);
        cur_intersection_size = new_size;
    }
    float iou = 0.;
    if (cur_intersection_size <= 2)
        return 0.;
    for (int i = 0; i < cur_intersection_size; ++i) {
        float x1 = intersection[i * 2 + 0];
        float y1 = intersection[i * 2 + 1];
        float x2 = intersection[(i + 1) % cur_intersection_size * 2 + 0];
        float y2 = intersection[(i + 1) % cur_intersection_size * 2 + 1];
        iou += diff_of_products<float>(x1, y2, y1, x2);
    }
    return iou * 0.5;

}

template<typename scalar_t>
__global__ void rotate_iou(const scalar_t* input1, const scalar_t* input2
        ,scalar_t* iou_output, const int size1, const int size2) {

    //int i = threadIdx.x;
    //int j = threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("fk me");
    //printf("threadid %d, %d, %d, %d, %d, %d ", i, j, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
    if (i >= size1)
        return;
    if (j >= size2)
        return;

    //for(int k = 0; k < 16; ++k) {

    //    printf("threadid %d, %d, %lf ", i, j, input1[k]);
    //}
    const scalar_t* rect1 = &input1[i * 4 * 2];
    const scalar_t* rect2 = &input2[j * 4 * 2];
    //printf("threadid %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf \n",
    // i, j, rect1[0], rect1[1], rect1[2], rect1[3], rect1[4], rect1[5], rect1[6], rect1[7]);
    // note the length is 8

    float intersec1 = intersection_area(rect1, rect1);
    float intersec2 = intersection_area(rect2, rect2);
    float intersec = intersection_area(rect1, rect2);

    float union_value = intersec1 + intersec2 - intersec;
    if (union_value != 0)
        iou_output[i * size2 + j] = intersec / (intersec1 + intersec2 - intersec);

}


template<typename scalar_t>
__global__ void rotate_iou_flag(const scalar_t* input1, const scalar_t* input2
        ,scalar_t* iou_output, const int size1, const int size2, const int32_t* flag) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= size1)
        return;
    if (j >= size2)
        return;

    if (flag[i * size2 + j] != i)
        return;
    const scalar_t* rect1 = &input1[i * 4 * 2];
    const scalar_t* rect2 = &input2[j * 4 * 2];
    //printf("threadid %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf \n",
    // i, j, rect1[0], rect1[1], rect1[2], rect1[3], rect1[4], rect1[5], rect1[6], rect1[7]);
    // note the length is 8

    float intersec1 = intersection_area(rect1, rect1);
    float intersec2 = intersection_area(rect2, rect2);
    float intersec = intersection_area(rect1, rect2);

    float union_value = intersec1 + intersec2 - intersec;
    if (union_value != 0)
        iou_output[i * size2 + j] = intersec / (intersec1 + intersec2 - intersec);

}


}
at::Tensor iou_cuda_forward(at::Tensor input1, at::Tensor input2) {
  int size1 = input1.size(0);
  int size2 = input2.size(0);
  //std::cout << " size " << input1.numel() << std::endl;
  at::Tensor output_iou =
      at::zeros({size1*size2}, input1.options().dtype(at::kFloat));
  AT_ASSERTM(output_iou.type().is_cuda(), "output_iou must be a CPU tensor");

  int block_size_x = size1 > 32? 32: size1;
  int block_size_y = size2 > 32? 32: size2;
  dim3 block(block_size_x, block_size_y);
  int grid_size_x = size1 / 32 + 1;
  int grid_size_y = size2 / 32 + 1;
  dim3 grid(grid_size_x, grid_size_y);
  //std::cout << grid_size_x << " " << grid_size_y << " " << block_size_x << " " << block_size_y << std::endl;

//  std::cout << "hit" << std::endl;
//  cudaDeviceSynchronize();

  AT_DISPATCH_FLOATING_TYPES(input1.type(), "iou_forward_cuda", ([&] {
    rotate_iou<scalar_t><<<grid, block>>>(
        input1.data<scalar_t>(),
        input2.data<scalar_t>(),
        output_iou.data<scalar_t>(),
        size1,
        size2
        );
  }));

//  cudaDeviceSynchronize();
  return output_iou;
}

at::Tensor iou_cuda_forward_with_flag(at::Tensor input1, at::Tensor input2,
                                      at::Tensor flag) {
  int size1 = input1.size(0);
  int size2 = input2.size(0);
  at::Tensor output_iou =
      at::zeros({size1*size2}, input1.options().dtype(at::kFloat));
  AT_ASSERTM(output_iou.type().is_cuda(), "output_iou must be a GPU tensor");

  int block_size_x = size1 > 32? 32: size1;
  int block_size_y = size2 > 32? 32: size2;
  dim3 block(block_size_x, block_size_y);
  int grid_size_x = size1 / 32 + 1;
  int grid_size_y = size2 / 32 + 1;
  dim3 grid(grid_size_x, grid_size_y);


  AT_DISPATCH_FLOATING_TYPES(input1.type(), "iou_forward_cuda", ([&] {
    rotate_iou_flag<scalar_t><<<grid, block>>>(
        input1.data<scalar_t>(),
        input2.data<scalar_t>(),
        output_iou.data<scalar_t>(),
        size1,
        size2,
        flag.data<int32_t>()
        );
  }));

  return output_iou;
}