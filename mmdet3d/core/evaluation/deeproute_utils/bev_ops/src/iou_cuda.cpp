#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor iou_cuda_forward(
    at::Tensor input1,
    at::Tensor input2
    );

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor iou_forward(
    at::Tensor input1,
    at::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  return iou_cuda_forward(input1, input2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iou_forward", &iou_forward, "iou forward (CUDA)");
//  m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}