#include "ops.h"
#include "tensor.h"
#include <cassert>
#include <fstream>
#include <stdexcept>

Tensor matmul(const Tensor &A, const Tensor &B) {
  Tensor result{std::vector<float>(A.row() * B.column(), 0),
                {A.shape[0], B.shape[1]}};
  for (int i{0}; i < A.row(); ++i) {
    for (int j{0}; j < B.column(); ++j) {
      for (int k{0}; k < A.column(); ++k) {
        result.at(i, j) += A.at(i, k) * B.at(k, j);
      }
    }
  }

  return result;
}

Tensor matadd(const Tensor &A, const Tensor &B) {
  assert(A.shape == B.shape);

  Tensor result{std::vector<float>(A.data.size()), A.shape};
  for (int i{0}; i < A.data.size(); ++i) {
    result.data[i] = A.data[i] + B.data[i];
  }
  return result;
}

Tensor SiLU(const Tensor &A) {
  Tensor result = A;
  for (int i{0}; i < result.data.size(); ++i) {
    result.data[i] = result.data[i] / (1 + std::exp(-result.data[i]));
  }
  return result;
}

void softmax(Tensor &A) {
  for (int i{0}; i < A.row(); ++i) {
    float exp_sum{0};
    float max_num =
        *std::max_element(A.data.begin() + (i * A.column()),
                          A.data.begin() + (i * A.column()) + A.column());
    for (int j{0}; j < A.column(); ++j) {
      exp_sum += std::exp(A.at(i, j) - max_num);
    }
    for (int k{0}; k < A.column(); ++k) {
      A.at(i, k) = std::exp(A.at(i, k) - max_num) / exp_sum;
    }
  }
}

Tensor transpose(const Tensor &A) {
  Tensor result{std::vector<float>(A.data.size(), 0), {A.column(), A.row()}};
  for (int i{0}; i < result.row(); ++i) {
    for (int j{0}; j < result.column(); ++j) {
      result.at(i, j) = A.at(j, i);
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
    float rms{std::sqrt(mean + 1e-5f)};
    for (int j{0}; j < T.column(); ++j) {
      T.at(i, j) = (T.at(i, j) / rms) * W.at(0, j);
    };
  }
}

Config readHeader(std::ifstream &file) {
  Config header;
  static_assert(sizeof(header) == 28, "Header must containt 7 int ");
  file.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!file)
    throw std::runtime_error("Error reading the error");
  return header;
}
