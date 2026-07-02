#include "iostream"
#include <vector>

using Matrix = std::vector<std::vector<int>>;

void matmul(Matrix a, Matrix b) {
  Matrix result(a.size(), std::vector<int>(b[0].size(), 0));
  for (int i{0}; i < a.size(); ++i) {
    for (int j{0}; j < b[0].size(); ++j) {
      for (int k{0}; k < a[0].size(); ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  for (std::vector<int> row : result) {
    for (int value : row) {
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
}

int main() {
  Matrix matrix_a{{1, 2, 3}, {4, 5, 6}};
  Matrix matrix_b{{7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
  matmul(matrix_a, matrix_b);

  return 0;
}
