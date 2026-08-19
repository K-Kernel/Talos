#include "ops.hpp"
#include "tensor.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

Tensor matvec(const Tensor &A, const Tensor &W) {
  assert(static_cast<int>(A.data.size()) == W.column());
  Tensor result{std::vector<float>(W.row(), 0), {1, W.row()}};

#pragma omp parallel for
  for (int i = 0; i < W.row(); ++i) {
    float sum{0};
    for (int k{0}; k < W.column(); ++k) {
      sum += W.at(i, k) * A.data[k];
    }
    result.data[i] = sum;
  }

  return result;
}
Tensor matadd_elementwise(const Tensor &A, const Tensor &B) {
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

  for (int i{0}; i < head_size; i += 2) {
    int idx = head_start + i;
    int pair_index{i / 2};
    float frequency{1.0f /
                    std::pow(10000.0f, (float)(2 * pair_index) / head_size)};
    float theta{position * frequency};

    float x = buffer[idx] * std::cos(theta) - buffer[idx + 1] * std::sin(theta);
    float y = buffer[idx] * std::sin(theta) + buffer[idx + 1] * std::cos(theta);

    buffer[idx] = x;
    buffer[idx + 1] = y;
  }
};

Tensor matmul_elementwise(const Tensor &A, const Tensor &B) {
  assert(A.shape == B.shape);
  Tensor result{std::vector<float>(A.data.size()), A.shape};
  for (size_t i = 0; i < A.data.size(); ++i)
    result.data[i] = A.data[i] * B.data[i];
  return result;
}

Tensor foward(int token, int pos, const transformerWeight &weight,
              const Config &config, std::vector<Tensor> &key_cache,
              std::vector<Tensor> &value_cache) {

  int head_size{config.dim / config.n_heads};
  Tensor T;
  T.data = embLookup(token, weight.emb, config.dim);
  T.shape = {1, config.dim};

  for (int layer{0}; layer < config.n_layers; ++layer) {
    Tensor T_prime{T};
    rmsnorm(T_prime, weight.rms_att_weight[layer]);

    Tensor q = matvec(T_prime, weight.wq[layer]);
    Tensor k = matvec(T_prime, weight.wk[layer]);
    Tensor v = matvec(T_prime, weight.wv[layer]);

    for (int h{0}; h < config.n_heads; ++h) {
      RoPE(q.data, head_size * h, head_size, pos);
      RoPE(k.data, head_size * h, head_size, pos);
    };

    for (int d{0}; d < config.dim; ++d) {
      key_cache[layer].at(pos, d) = k.data[d];
      value_cache[layer].at(pos, d) = v.data[d];
    }

    Tensor output{std::vector<float>(config.dim, 0), {1, config.dim}};

    for (int h{0}; h < config.n_heads; ++h) {
      Tensor scores{std::vector<float>(pos + 1, 0), {1, pos + 1}};
      for (int p{0}; p <= pos; ++p) {
        float dot{0};
        for (int i{0}; i < head_size; ++i) {
          dot += q.data[h * head_size + i] *
                 key_cache[layer].at(p, h * head_size + i);
        }
        scores.data[p] = dot / std::sqrt(head_size);
      }
      softmax(scores);
      for (int d{0}; d < head_size; ++d) {
        float sum{0};
        for (int p{0}; p <= pos; ++p) {
          sum += scores.data[p] * value_cache[layer].at(p, h * head_size + d);
        }
        output.data[h * head_size + d] = sum;
      }
    }
    Tensor att_out = matvec(output, weight.wo[layer]);
    T = matadd_elementwise(T, att_out);

    Tensor T_ffn{T};
    rmsnorm(T_ffn, weight.rms_fnn_weight[layer]);

    Tensor gate = matvec(T_ffn, weight.w1[layer]);
    Tensor up = matvec(T_ffn, weight.w3[layer]);

    gate = SiLU(gate);
    Tensor gated{matmul_elementwise(gate, up)};

    Tensor ffn_out = matvec(gated, weight.w2[layer]);
    T = matadd_elementwise(T, ffn_out);
  }

  rmsnorm(T, weight.rms_final_weight);
  return matvec(T, weight.emb); // logits tensor
}
