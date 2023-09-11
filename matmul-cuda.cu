#include <iostream>
#include <ranges>

#include <cublas_v2.h>
#include <eigen3/Eigen/Dense>
#include <thrust/device_vector.h>

#include "utility.h"

constexpr auto M = 1'001;
constexpr auto N = 1'002;
constexpr auto K = 1'003;

auto matmul(cublasHandle_t &handle, float *x_, float *y_, float *z_, int m, int n, int k) {
  const auto alpha = 1.0f;
  const auto beta  = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, x_, m, y_, k, &beta, z_, m);  // GEneral Matrix-Matrix multiplication。
}

int main(int argc, char** argv) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  srand(0);

  const auto x = static_cast<Eigen::MatrixXf>(Eigen::MatrixXf::Random(M, K));
  const auto y = static_cast<Eigen::MatrixXf>(Eigen::MatrixXf::Random(K, N));

  auto x_ = thrust::device_vector<float>(M * K);
  auto y_ = thrust::device_vector<float>(K * N);

  cublasSetMatrix(M, K, sizeof(float), x.transpose().data(), M, x_.data().get(), M);  // Eigenはrow-majorにも対応しているのでこのやり方は無駄なのだけど、とりあえずtranspose()で。
  cublasSetMatrix(K, N, sizeof(float), y.transpose().data(), K, y_.data().get(), K);

  cudaDeviceSynchronize();

  for (const auto &i : std::views::iota(0, 6)) {
    auto z_ = thrust::device_vector<float>(M * N, 0.0f);

    const auto &[duration, _] = duration_and_result([&] {
      matmul(handle, x_.data().get(), y_.data().get(), z_.data().get(), M, N, K);

      cudaDeviceSynchronize();

      return 0;
    });

    if (i == 0) {
      continue;
    }

    cudaDeviceSynchronize();

    std::cout << duration << "\t" << z_[0] << "\t" << z_[1] << std::endl;
  }

  cublasDestroy(handle);

  return 0;
}
