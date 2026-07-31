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

Tensor readTensor(std::ifstream &file, std::vector<int> Tshape) {
  Tensor t;
  t.shape = Tshape;

  size_t n = 1;
  for (int x : Tshape) {
    n *= x;
  }
  t.data.resize(n);

  file.read(reinterpret_cast<char *>(t.data.data()), n * sizeof(float));
  if (!file) {
    throw std::runtime_error("Couldn't read the file");
  }
  return t;
}

transformerWeight weightLoader(std::ifstream &file, Config &headerConfig) {
  int head_size{headerConfig.dim / headerConfig.n_heads};
  int L = headerConfig.n_layers;
  transformerWeight weight(L);

  weight.emb = readTensor(file, {headerConfig.vocab_size, headerConfig.dim});
  for (int i{0}; i < L; ++i) {
    weight.rms_att_weight[i] = readTensor(file, {1, headerConfig.dim});
  };
  for (int i{0}; i < L; ++i) {
    weight.wq[i] = readTensor(file, {headerConfig.dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    weight.wk[i] = readTensor(
        file, {headerConfig.dim, headerConfig.n_kv_heads * head_size});
  }
  for (int i{0}; i < L; ++i) {
    weight.wv[i] = readTensor(
        file, {headerConfig.dim, headerConfig.n_kv_heads * head_size});
  }
  for (int i{0}; i < L; ++i) {
    weight.wo[i] = readTensor(file, {headerConfig.dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    weight.rms_fnn_weight[i] = readTensor(file, {1, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    weight.w1[i] =
        readTensor(file, {headerConfig.hidden_dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    weight.w2[i] =
        readTensor(file, {headerConfig.dim, headerConfig.hidden_dim});
  }
  for (int i{0}; i < L; ++i) {
    weight.w3[i] =
        readTensor(file, {headerConfig.hidden_dim, headerConfig.dim});
  }

  weight.rms_final_weight = readTensor(file, {1, headerConfig.dim});
  Tensor freq_real = readTensor(file, {headerConfig.seq_len, head_size / 2});
  Tensor freq_imag = readTensor(file, {headerConfig.seq_len, head_size / 2});

  return weight;
};

std::vector<float> embLookup(unsigned int token_id, const Tensor &embedding,
                             int dim) {
  auto start = embedding.data.begin() + (token_id * dim);
  return std::vector<float>(start, start + dim);
};

void RoPE(std::vector<float> &buffer, int head_start, int head_size,
          int position) {
  if (!(head_size % 2 == 0)) {
    throw std::runtime_error("The tensor is not even");
  };

  for (size_t i{0}; i < head_size; i += 2) {
    int idx = head_start + i;
    int pair_index{static_cast<int>(i) / 2};
    float frequency{1.0f /
                    std::pow(10000.0f, (float)(2 * pair_index) / head_size)};
    float theta{position * frequency};

    float x = buffer[idx] * std::cos(theta) - buffer[i + 1] * std::sin(theta);
    float y = buffer[idx] * std::sin(theta) + buffer[i + 1] * std::cos(theta);

    buffer[i] = x;
    buffer[i + 1] = y;
  }
};
