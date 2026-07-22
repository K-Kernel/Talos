#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  int row() const { return shape[0]; };
  int column() const { return shape[1]; };

  // one is for read purposes only and the other is for write
  float at(int r, int c) const { return data[r * column() + c]; }
  float &at(int r, int c) { return data[r * column() + c]; }
};

Tensor matmul(const Tensor &T, const Tensor &b) {
  Tensor result{std::vector<float>(T.row() * b.column(), 0),
                {T.shape[0], b.shape[1]}};
  for (int i{0}; i < T.row(); ++i) {
    for (int j{0}; j < b.column(); ++j) {
      for (int k{0}; k < T.column(); ++k) {
        result.at(i, j) += T.at(i, k) * b.at(k, j);
      }
    }
  }

  return result;
}

Tensor matadd(const Tensor &T, const Tensor &b) {
  assert(T.shape == b.shape);

  Tensor result{std::vector<float>(T.data.size()), T.shape};
  for (int i{0}; i < T.data.size(); ++i) {
    result.data[i] = T.data[i] + b.data[i];
  }
  return result;
}

Tensor SiLU(const Tensor &T) {
  Tensor result = T;
  for (int i{0}; i < result.data.size(); ++i) {
    result.data[i] = result.data[i] / (1 + std::exp(-result.data[i]));
  }
  return result;
}

void softmax(Tensor &T) {
  for (int i{0}; i < T.row(); ++i) {
    float exp_sum{0};
    float max_num =
        *std::max_element(T.data.begin() + (i * T.column()),
                          T.data.begin() + (i * T.column()) + T.column());
    for (int j{0}; j < T.column(); ++j) {
      exp_sum += std::exp(T.at(i, j) - max_num);
    }
    for (int k{0}; k < T.column(); ++k) {
      T.at(i, k) = std::exp(T.at(i, k) - max_num) / exp_sum;
    }
  }
}

Tensor transpose(const Tensor &T) {
  Tensor result{std::vector<float>(T.data.size(), 0), {T.column(), T.row()}};
  for (int i{0}; i < result.row(); ++i) {
    for (int j{0}; j < result.column(); ++j) {
      result.at(i, j) = T.at(j, i);
    }
  };
  return result;
}

void rmsnorm(Tensor &T, const Tensor &W) {
  for (int i{0}; i < T.row(); ++i) {
    float mean{0};
    for (int j{0}; j < T.column(); ++j) {
      mean += T.at(i, j) * T.at(i, j) / T.column();
    }
    float rms{std::sqrt(mean) + 1e-5f};
    for (int j{0}; j < T.column(); ++j) {
      T.at(i, j) = (T.at(i, j) / rms) * W.data[j];
    };
  }
}

int main() {
  Tensor tensor_test{{101, 102, 103, 104, 105, 106}, {2, 3}};
  transpose(tensor_test);
  Tensor tensor_test2{{1, 2, 3, 4}, {2, 2}};
  Tensor weight{{1, 1, 1, 1}, {1, 4}};
  rmsnorm(tensor_test2, weight);
}
