#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>

#include "utility.h"

auto get_pi(int n, int seed) {
  auto rng = std::default_random_engine(seed);

  const auto c = std::ranges::count_if(std::views::zip(std::views::iota(0, n) | std::views::transform([&](const auto &_) { return std::uniform_real_distribution(0.0f, 1.0f)(rng); }),
                                                       std::views::iota(0, n) | std::views::transform([&](const auto &_) { return std::uniform_real_distribution(0.0f, 1.0f)(rng); })),
                                       [](const auto &p) {
                                         return std::pow(std::get<0>(p), 2) + std::pow(std::get<1>(p), 2) <= 1.0f;
                                       });

  // const auto ps = std::views::zip(std::views::iota(0, n) | std::views::transform([&](const auto &_) { return std::uniform_real_distribution(0.0f, 1.0f)(rng); }),
  //                                 std::views::iota(0, n) | std::views::transform([&](const auto &_) { return std::uniform_real_distribution(0.0f, 1.0f)(rng); }));

  // const auto c = std::count_if(std::execution::par_unseq,
  //                              std::begin(ps),
  //                              std::end(ps),
  //                              [](const auto &p) {
  //                                return std::pow(std::get<0>(p), 2) + std::pow(std::get<1>(p), 2) <= 1.0f;
  //                              });

  return 4.0f * static_cast<float>(c) / static_cast<float>(n);  // c / n = pi * 1 * 1 / 4。だから4 * c / n = pi。
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
