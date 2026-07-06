#include "iostream"
#include <vector>

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  int row() const { return shape[0]; };
  int column() const { return shape[1]; };
  float &at(int r, int c) { return data[r * column() + c]; }
};

Tensor matmul(Tensor &a, Tensor &b) {
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

int main() {
  Tensor matrix_a{{1, 2, 3, 4, 5, 6}, {2, 3}};
  Tensor matrix_b{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4}};
  auto resultTensor = matmul(matrix_a, matrix_b);

  for (int i{0}; i < resultTensor.row(); ++i) {
    for (int j{0}; j < resultTensor.column(); ++j) {
      std::cout << resultTensor.at(i, j) << " ";
    }
    std::cout << '\n';
  }

  return 0;
}
