#include <iostream>
#include <ranges>

#include <eigen3/Eigen/Dense>

#include "utility.h"

constexpr auto M = 1'001;
constexpr auto N = 1'002;
constexpr auto K = 1'003;

auto matmul(const Eigen::MatrixXf &x, const Eigen::MatrixXf &y) {
  return static_cast<Eigen::MatrixXf>(x * y);  // Eigenは型が確定するまで評価を遅延させるので、static_castして評価させます。
}

int main(int argc, char** argv) {
  srand(0);

  const auto x = static_cast<Eigen::MatrixXf>(Eigen::MatrixXf::Random(M, K));
  const auto y = static_cast<Eigen::MatrixXf>(Eigen::MatrixXf::Random(K, N));

  for (const auto &i : std::views::iota(0, 6)) {
    const auto &[duration, result] = duration_and_result([&] {
      return matmul(x, y);
    });

    if (i == 0) {
      continue;
    }

    std::cout << duration << "\t" << result(0, 0) << "\t" << result(1, 0) << std::endl;
  }

  return 0;
}
