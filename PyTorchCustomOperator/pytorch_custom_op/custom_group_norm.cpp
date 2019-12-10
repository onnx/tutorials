#include <torch/script.h>
#include "Eigen/Dense"

using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;

torch::Tensor custom_group_norm(torch::Tensor X, torch::Tensor num_groups, torch::Tensor scale, torch::Tensor bias, double eps) {

  float* X_data = X.data<float>();
  float* scale_data = scale.data<float>();
  float* bias_data = bias.data<float>();
  int num_groups_i = int(num_groups.data<float>()[0]);
  torch::Tensor output = torch::zeros(X.sizes());
  float* out = output.data<float>();
  const int64_t N = X.size(0);
  const int64_t C = X.size(1) / num_groups_i;  // assume [N C*num_groups H W]  per the spec

  // Do computation
  int64_t sample_size = 1;
  for (auto i = 2; i < X.dim(); ++i) {
    sample_size *= X.size(i);
  }
  sample_size *= C;

  for (auto i = 0; i < N * num_groups_i; ++i) {
    ConstEigenVectorArrayMap Xi(X_data + sample_size * i, sample_size);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.f / std::sqrt(squared_norm / sample_size + eps);
    EigenVectorArrayMap Yi(out + sample_size * i, sample_size);
    const float channel_scale = inv_stdev * scale_data[i % (C * num_groups_i)];
    const float channel_shift = bias_data[i % (C * num_groups_i)] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
   }

  return output.clone();
}

static auto registry =
  torch::RegisterOperators("mynamespace::custom_group_norm", &custom_group_norm);
