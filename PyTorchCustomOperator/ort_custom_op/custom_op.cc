/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "Eigen/Dense"
#include "onnxruntime_cxx_api.h"

template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
void GroupNormKernel<T>::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
  const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
  const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));
  const OrtValue* input_scale = ort_.KernelContext_GetInput(context, 2);
  const T* scale_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_scale));
  const OrtValue* input_B = ort_.KernelContext_GetInput(context, 3);
  const T* B_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_B));

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);
  const int64_t N = dimensions[0];
  const int64_t C = dimensions[1] / num_groups[0];  // assume [N C*num_groups H W]  per the spec

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
	int64_t sample_size = 1;
  for (size_t i = 2; i < dimensions.size(); ++i) {
    sample_size *= dimensions[i];
  }
  sample_size *= C;

  for (auto i = 0; i < N * num_groups[0]; ++i) {
    ConstEigenVectorArrayMap<float> Xi(X_data + sample_size * i, sample_size);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.0f / std::sqrt(squared_norm / sample_size + epsilon_);
    EigenVectorArrayMap<float> Yi(out + sample_size * i, sample_size);
    const float channel_scale = inv_stdev * scale_data[i % (C * int(num_groups[0]))];
    const float channel_shift = B_data[i % (C * int(num_groups[0]))] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }
}
