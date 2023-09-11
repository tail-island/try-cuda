#include <chrono>

template <typename T>
auto duration_and_result(const T f) {
  const auto starting_time = std::chrono::high_resolution_clock::now();

  const auto result = f();

  return std::make_tuple(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - starting_time),
                         result);
}

