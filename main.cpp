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

Tensor matmul(const Tensor &a, const Tensor &b) {
  Tensor result{std::vector<float>(a.row() * b.column(), 0),
                {a.shape[0], b.shape[1]}};
  for (int i{0}; i < a.row(); ++i) {
    for (int j{0}; j < b.column(); ++j) {
      for (int k{0}; k < a.column(); ++k) {
        result.at(i, j) += a.at(i, k) * b.at(k, j);
      }
    }
  }

  return result;
}

Tensor matadd(const Tensor &a, const Tensor &b) {
  assert(a.shape == b.shape);

  Tensor result{std::vector<float>(a.data.size()), a.shape};
  for (int i{0}; i < a.data.size(); ++i) {
    result.data[i] = a.data[i] + b.data[i];
  }
  return result;
}

Tensor SiLU(const Tensor &a) {
  Tensor result = a;
  for (int i{0}; i < result.data.size(); ++i) {
    result.data[i] = result.data[i] / (1 + std::exp(-result.data[i]));
  }
  return result;
}

void softmax(Tensor &a) {
  for (int i{0}; i < a.row(); ++i) {
    float exp_sum{0};
    float max_num =
        *std::max_element(a.data.begin() + (i * a.column()),
                          a.data.begin() + (i * a.column()) + a.column());
    for (int j{0}; j < a.column(); ++j) {
      exp_sum += std::exp(a.at(i, j) - max_num);
    }
    for (int k{0}; k < a.column(); ++k) {
      a.at(i, k) = std::exp(a.at(i, k) - max_num) / exp_sum;
    }
  }
}

Tensor transpose(const Tensor &T) {
  Tensor result{std::vector<float>(T.data.size(), 0), {T.column(), T.row()}};
  for (int i{0}; i < T.row(); ++i) {
    for (int j{0}; j < T.column(); ++j) {
      result.at(i, j) = T.at(j, i);
    }
  };
  for (float x : result.data) {
    std::cout << x << '\n';
  }
  return result;
};

int main() {
  Tensor tensor_test{{101, 102, 103, 104, 105, 106}, {2, 3}};
  transpose(tensor_test);
}
