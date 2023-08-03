#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <tuple>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
auto duration_and_result(const T f) {
  const auto starting_time = std::chrono::high_resolution_clock::now();

  const auto result = f();

  return std::make_tuple(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - starting_time),
                         result);
}

auto step_0(int *xs, int n) {
  for (auto i = 1; i < n; ++i) {
    xs[0] += xs[i];
  }
}

__global__
void step_1(int *xs, int n) {
  for (auto i = 1; i < n; ++i) {
    xs[0] += xs[i];
  }
}

int main(int argc, char **argv) {
  auto rng = std::default_random_engine(0);

  const auto numbers = [&] {
    auto result = std::vector<int>();

    result.reserve(1'000'000);

    for (const auto &_ : std::views::iota(0, 1'000'000)) {
      result.emplace_back(std::uniform_int_distribution<int>(0, 2)(rng));
    }

    return result;
  }();

  [&] {
    auto numbers_ = thrust::host_vector<int>(std::begin(numbers), std::end(numbers));

    const auto &[duration, result] = duration_and_result([&] {
      step_0(numbers_.data(), std::size(numbers_));
      return numbers_[0];
    });

    std::cout << "step 0: " << duration << "\t" << result << std::endl;
  }();

  [&] {
    auto numbers_ = thrust::device_vector<int>(std::begin(numbers), std::end(numbers));
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      step_1<<<1, 1>>>(numbers_.data().get(), std::size(numbers_));
      const auto result = numbers_[0];
      cudaDeviceSynchronize();

      return result;
    });

    std::cout << "step 1: " << duration << "\t" << result << std::endl;
  }();

  [&] {
    auto numbers_ = thrust::device_vector<int>(std::begin(numbers), std::end(numbers));
    cudaDeviceSynchronize();

    const auto &[duration, result] = duration_and_result([&] {
      const auto result = thrust::reduce(std::begin(numbers_), std::end(numbers_), 0, thrust::plus<int>());
      cudaDeviceSynchronize();

      return result;
    });

    std::cout << "step x: " << duration << "\t" << result << std::endl;
  }();

  [&] {
    const auto &[duration, result] = duration_and_result([&] {
      return std::reduce(std::begin(numbers), std::end(numbers), 0);
    });

    std::cout << "step y: " << duration << "\t" << result << std::endl;
  }();

  [&] {
    const auto &[duration, result] = duration_and_result([&] {
      return std::reduce(std::execution::par_unseq, std::begin(numbers), std::end(numbers), 0);
    });

    std::cout << "step z: " << duration << "\t" << result << std::endl;
  }();

  return 0;
}
