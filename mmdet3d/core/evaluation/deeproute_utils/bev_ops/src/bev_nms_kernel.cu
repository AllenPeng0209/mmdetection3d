#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ float intersect_areaf(float const * const rect1, float const * const rect2) {
	float intersection[200 * 2] = {0};
	float line_v[200] = {0};
	float new_intersec[200 * 2] = {0};
	// I use loop based method, it is not elegant
	// TODO need to fix
//	cudaMemcpy(&intersection, rect1, sizeof(float) * 8, cudaMemcpyDeviceToDevice);

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

		 // max size assume


		float a = y2 - y1;
		float b = x1 - x2;
		float c = x2 * y1 - y2 * x1;

		for (int j = 0; j < cur_intersection_size; ++j) {
			float c_x = intersection[j * 2 + 0];
			float c_y = intersection[j * 2 + 1];
			line_v[j] = a * c_x + b * c_y + c;
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
				float c1 = x4 * y3 - y4 * x3;
				float w = a * b1 - b * a1;
				float c_x = (b * c1 - c * b1) / w;
			    float c_y = (c * a1 - a * c1) / w;
				new_intersec[2 * new_size + 0] = c_x;
				new_intersec[2 * new_size + 1] = c_y;
				new_size += 1;

			}
		}
		for (int k = 0; k < new_size; k++) {
			intersection[2 * k + 0] = new_intersec[2 * k + 0];
			intersection[2 * k + 1] = new_intersec[2 * k + 1];
		}
//		cudaMemcpy(&intersection, &new_intersec, sizeof(float) * new_size * 2, cudaMemcpyDeviceToDevice);
		cur_intersection_size = new_size;
	}
	float iou = 0.;
	for (int i = 0; i < cur_intersection_size; ++i) {
	    float x1 = intersection[i * 2 + 0];
	    float y1 = intersection[i * 2 + 1];
	    float x2 = intersection[(i + 1) % cur_intersection_size * 2 + 0];
	    float y2 = intersection[(i + 1) % cur_intersection_size * 2 + 1];
	    iou += (x1 * y2 - y1 * x2);
	}
	return iou * 0.5;

}

__global__ void nms_kernel(const int n_boxes, const float* nms_overlap_thresh,
                           const float *corners, const float* category,
                           const float* scores,
                           unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  __shared__ float block_boxes[threadsPerBlock * 10]; // for boxes with scores and category
//  __shared__ float block_scores[threadsPerBlock * 1]; // for scores

  if (threadIdx.x < col_size) {
	  for (int i = 0; i < 8; ++i) {
		  // for each batch copy to shared_memory
		  //printf("setting %d %lf ", threadIdx.x, corners[(threadsPerBlock * col_start + threadIdx.x) * 8 + i]);
		  block_boxes[threadIdx.x * 10 + i] = corners[(threadsPerBlock * col_start + threadIdx.x) * 8 + i];
	  }
	  //printf("setting %d %lf ", threadIdx.x, scores[(threadsPerBlock * col_start + threadIdx.x)]);
	  block_boxes[threadIdx.x * 10 + 8] = scores[(threadsPerBlock * col_start + threadIdx.x)];
	  block_boxes[threadIdx.x * 10 + 9] = category[(threadsPerBlock * col_start + threadIdx.x)];
  }
  // at thread level
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float* cur_box = &corners[cur_box_idx * 8];
    const float* cur_score = &scores[cur_box_idx];
    const float* cur_cat = &category[cur_box_idx];
    const int type = int(cur_cat[0]);
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    // for the same object pass
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    float area1 = intersect_areaf(cur_box, cur_box);
    for (i = start; i < col_size; i++) {
      float query_area = intersect_areaf(block_boxes + i * 10, block_boxes + i * 10);
      float intersec = intersect_areaf(cur_box, block_boxes + i * 10);
      float iou = intersec / (area1 + query_area - intersec + 1e-8);
      //printf("iou %lf, %lf, %lf \n", iou, cur_score[0], block_boxes[i * 10 + 8]);
      if (iou > nms_overlap_thresh[type] && (cur_score[0] > block_boxes[i * 10 + 8])) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

at::Tensor bev_nms_cuda(const at::Tensor& locs, const at::Tensor& scores, const at::Tensor& classes,
                        const at::Tensor& threshold) {
  using scalar_t = float;
  AT_ASSERTM(locs.type().is_cuda(), "boxes must be a CUDA tensor");
  //std::cout << locs.device().index() << std::endl;
  THCudaCheck(cudaSetDevice(int(locs.device().index())));

  int boxes_num = locs.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = locs.data<scalar_t>();
  scalar_t* scores_dev = scores.data<scalar_t>();
  // todo fix the type later
  scalar_t* type_dev = classes.data<scalar_t>();
  scalar_t* thresh_dev = threshold.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState
  //std::cout << "he1" << std::endl;

  unsigned long long* mask_dev = NULL;

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  //std::cout << "here 2 " << std::endl;
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  thresh_dev,
                                  boxes_dev,
                                  type_dev,
                                  scores_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, locs.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                     locs.device(), keep.scalar_type());
}