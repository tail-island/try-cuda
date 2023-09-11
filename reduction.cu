#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <thread>
#include <tuple>

#include <cooperative_groups.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utility.h"

const auto CUDA_THREAD_SIZE = 256;

// CPUで1並列で実行します。

auto step_0_kernel(int *xs, int xs_size) {
  for (auto i = 1; i < xs_size; ++i) {
    xs[0] += xs[i];
  }
}

auto step_0(thrust::host_vector<int> &xs) {
  step_0_kernel(xs.data(), std::size(xs));

  return xs[0];
}

// GPUで1並列で実行します。

__global__
void step_1_kernel(int *xs_, int xs_size) {
  for (auto i = 1; i < xs_size; ++i) {
    xs_[0] += xs_[i];
  }
}

auto step_1(thrust::device_vector<int> &xs_) {
  step_1_kernel<<<1, 1>>>(xs_.data().get(), std::size(xs_));

  cudaDeviceSynchronize();

  return xs_[0];
}

// 隣同士を足し合わせる形で、並列化します。

__global__
void step_2_kernel(int *xs_, int xs_size, int *ys_) {
  const auto i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const auto group = cooperative_groups::this_thread_block();

  for (auto j = 1; j < blockDim.x * 2; j <<= 1) {
    if (i % (j * 2) == 0 && i + j < xs_size) {
      xs_[i] += xs_[i + j];
    }
    cooperative_groups::sync(group);
  }

  if (threadIdx.x == 0) {
    ys_[blockIdx.x] = xs_[i];
  }
}

auto step_2(thrust::device_vector<int> &xs_) {
  for (;;) {
    const auto block_size = (std::size(xs_) / 2 + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

    auto ys_ = thrust::device_vector<int>(block_size);
    step_2_kernel<<<block_size, CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

    if (block_size == 1) {
      cudaDeviceSynchronize();

      return thrust::host_vector<int>{ys_}[0];
    }

    xs_ = std::move(ys_);
  }
}

// Shared Memoryを使用します。

__global__
void step_3_kernel(int *xs_, int xs_size, int *ys_) {
  extern __shared__ int shared_memory[];

  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto group = cooperative_groups::this_thread_block();

  shared_memory[threadIdx.x] = i < xs_size ? xs_[i] : 0;
  cooperative_groups::sync(group);

  for (auto j = 1; j < blockDim.x; j <<= 1) {
    if (i % (j * 2) == 0) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + j];
    }
    cooperative_groups::sync(group);
  }

  if (threadIdx.x == 0) {
    ys_[blockIdx.x] = shared_memory[threadIdx.x];
  }
}

auto step_3(thrust::device_vector<int> &xs_) {
  for (;;) {
    const auto block_size = (std::size(xs_) + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

    auto ys_ = thrust::device_vector<int>(block_size);
    step_3_kernel<<<block_size, CUDA_THREAD_SIZE, sizeof(int) * CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

    if (block_size == 1) {
      cudaDeviceSynchronize();

      return thrust::host_vector<int>{ys_}[0];
    }

    xs_ = std::move(ys_);
  }
}

// 条件をstep_2に揃えて、Shared Memoryを使用します。

// __global__
// void step_4_kernel(int *xs_, int xs_size, int *ys_) {
//   extern __shared__ int shared_memory[];

//   const auto i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
//   const auto group = cooperative_groups::this_thread_block();

//   shared_memory[threadIdx.x] = (i < xs_size ? xs_[i] : 0) + (i + 1 < xs_size ? xs_[i + 1] : 0);
//   cooperative_groups::sync(group);

//   for (auto j = 2; j < blockDim.x; j <<= 1) {
//     if (i % (j * 2) == 0) {
//       shared_memory[threadIdx.x] += shared_memory[threadIdx.x + j];
//     }
//     cooperative_groups::sync(group);
//   }

//   if (threadIdx.x == 0) {
//     ys_[blockIdx.x] = shared_memory[threadIdx.x];
//   }
// }

// auto step_4(thrust::device_vector<int> &xs_) {
//   for (;;) {
//     const auto block_size = (std::size(xs_) / 2 + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

//     auto ys_ = thrust::device_vector<int>(block_size);
//     step_4_kernel<<<block_size, CUDA_THREAD_SIZE, sizeof(int) * CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

//     if (block_size == 1) {
//       cudaDeviceSynchronize();

//       return thrust::host_vector<int>{ys_}[0];
//     }

//     xs_ = std::move(ys_);
//   }
// }

// 連続したスレッドを使用して、Warpのdivergenceを削減します。

__global__
void step_4_kernel(int *xs_, int xs_size, int *ys_) {
  extern __shared__ int shared_memory[];

  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto group = cooperative_groups::this_thread_block();

  shared_memory[threadIdx.x] = i < xs_size ? xs_[i] : 0;
  cooperative_groups::sync(group);

  for (auto j = 1; j < blockDim.x; j <<= 1) {
    const auto k = threadIdx.x * j * 2;
    if (k < blockDim.x) {
      shared_memory[k] += shared_memory[k + j];
    }
    cooperative_groups::sync(group);
  }

  if (threadIdx.x == 0) {
    ys_[blockIdx.x] = shared_memory[threadIdx.x];
  }
}

auto step_4(thrust::device_vector<int> &xs_) {
  for (;;) {
    const auto block_size = (std::size(xs_) + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

    auto ys_ = thrust::device_vector<int>(block_size);
    step_4_kernel<<<block_size, CUDA_THREAD_SIZE, sizeof(int) * CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

    if (block_size == 1) {
      cudaDeviceSynchronize();

      return thrust::host_vector<int>{ys_}[0];
    }

    xs_ = std::move(ys_);
  }
}

// 連続したメモリにアクセスして、Shared Memoryのbank conflictを削減します。

__global__
void step_5_kernel(int *xs_, int xs_size, int *ys_) {
  extern __shared__ int shared_memory[];

  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto group = cooperative_groups::this_thread_block();

  shared_memory[threadIdx.x] = i < xs_size ? xs_[i] : 0;
  cooperative_groups::sync(group);

  for (auto j = blockDim.x / 2; j > 0; j >>= 1) {
    if (threadIdx.x < j) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + j];
    }
    cooperative_groups::sync(group);
  }

  if (threadIdx.x == 0) {
    ys_[blockIdx.x] = shared_memory[threadIdx.x];
  }
}

auto step_5(thrust::device_vector<int> &xs_) {
  for (;;) {
    const auto block_size = (std::size(xs_) + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

    auto ys_ = thrust::device_vector<int>(block_size);
    step_5_kernel<<<block_size, CUDA_THREAD_SIZE, sizeof(int) * CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

    if (block_size == 1) {
      cudaDeviceSynchronize();

      return thrust::host_vector<int>{ys_}[0];
    }

    xs_ = std::move(ys_);
  }
}

// 休止してしまうスレッドを削減します。

__global__
void step_6_kernel(int *xs_, int xs_size, int *ys_) {
  extern __shared__ int shared_memory[];

  const auto i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  const auto group = cooperative_groups::this_thread_block();

  shared_memory[threadIdx.x] = (i < xs_size ? xs_[i] : 0) + (i + blockDim.x < xs_size ? xs_[i + blockDim.x] : 0);
  cooperative_groups::sync(group);

  for (auto j = blockDim.x / 2; j > 0; j >>= 1) {
    if (threadIdx.x < j) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + j];
    }
    cooperative_groups::sync(group);
  }

  if (threadIdx.x == 0) {
    ys_[blockIdx.x] = shared_memory[threadIdx.x];
  }
}

auto step_6(thrust::device_vector<int> &xs_) {
  for (;;) {
    const auto block_size = (std::size(xs_) / 2 + CUDA_THREAD_SIZE - 1) / CUDA_THREAD_SIZE;

    auto ys_ = thrust::device_vector<int>(block_size);
    step_6_kernel<<<block_size, CUDA_THREAD_SIZE, sizeof(int) * CUDA_THREAD_SIZE>>>(xs_.data().get(), std::size(xs_), ys_.data().get());

    if (block_size == 1) {
      cudaDeviceSynchronize();

      return thrust::host_vector<int>{ys_}[0];
    }

    xs_ = std::move(ys_);
  }
}

// テストします。

template <typename T>
auto test(T &rng, int numbers_size) {
  std::cout << numbers_size << std::endl;

  const auto numbers = [&] {
    auto result = std::vector<int>{};

    result.reserve(numbers_size);

    for (const auto &_ : std::views::iota(0, numbers_size)) {
      result.emplace_back(std::uniform_int_distribution<int>(0, 2)(rng));
    }

    return result;
  }();

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs = thrust::host_vector<int>{std::begin(numbers), std::end(numbers)};

    const auto &[duration, result] = duration_and_result([&] {
      return step_0(xs);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 0\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_1(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 1\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_2(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 2\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_3(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 3\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_4(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 4\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_5(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 5\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      return step_6(xs_);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step 6\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    auto xs_ = thrust::device_vector<int>{std::begin(numbers), std::end(numbers)};
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      const auto result = thrust::reduce(std::begin(xs_), std::end(xs_), 0, thrust::plus<int>{});
      cudaDeviceSynchronize();

      return result;
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step x\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    const auto &[duration, result] = duration_and_result([&] {
      return std::reduce(std::begin(numbers), std::end(numbers), 0);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step y\t" << duration << "\t" << result << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (const auto &i : std::views::iota(0, 6)) {
    const auto &[duration, result] = duration_and_result([&] {
      return std::reduce(std::execution::par_unseq, std::begin(numbers), std::end(numbers), 0);
    });

    if (i == 0) {
      continue;
    }

    std::cout << "step z:\t" << duration << "\t" << result << std::endl;
  }
}

int main(int argc, char **argv) {
  auto rng = std::default_random_engine(0);

  for (const auto &i : std::views::iota(1, 10 + 1)) {
    test(rng, i * 1'000'000);
  }

  return 0;
}
