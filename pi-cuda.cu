#include <iostream>
#include <ranges>

#include <curand.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include "utility.h"

float get_pi(int n, int seed) {
  auto rng = curandGenerator_t{};
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(rng, seed);

  auto xs = thrust::device_vector<float>(n);
  curandGenerateUniform(rng, xs.data().get(), n);

  auto ys = thrust::device_vector<float>(n);
  curandGenerateUniform(rng, ys.data().get(), n);

  auto it = thrust::make_zip_iterator(std::begin(xs), std::begin(ys));
  const auto c = thrust::count_if(it,
                                  it + n,
                                  [] __device__ (const auto &p) {
                                    return std::pow(thrust::get<0>(p), 2) + std::pow(thrust::get<1>(p), 2) <= 1.0f;
                                  });

  return 4.0f * static_cast<float>(c) / static_cast<float>(n);
}

int main(int argc, char **argv) {
  for (const auto &i : std::views::iota(0, 6)) {
    const auto &[duration, result] = duration_and_result([&] {
      return get_pi(10'000'000, i);
    });

    if (i == 0) {
      continue;
    }

    std::cout << duration << "\t" << result << std::endl;
  }

  return 0;
}
