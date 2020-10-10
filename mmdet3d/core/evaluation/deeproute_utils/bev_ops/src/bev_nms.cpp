#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor bev_nms_cuda(const at::Tensor& locs, const at::Tensor& scores, const at::Tensor& classes,
                        const at::Tensor& threshold);

at::Tensor bev_nms(const at::Tensor& locs, const at::Tensor& scores, const at::Tensor& classes,
                   const at::Tensor& threshold) {
  CHECK_CUDA(locs);
  if (locs.numel() == 0)
    return at::empty({0}, locs.options().dtype(at::kLong).device(at::kCPU));
  return bev_nms_cuda(locs, scores, classes, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_nms", &bev_nms, "non-maximum suppression");
}