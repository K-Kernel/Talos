#include "ops.h"
#include "tensor.h"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

struct transformerWeight {
  Tensor emb, rms_final_weight;
  std::vector<Tensor> rms_att_weight, wq, wk, wv, wo, rms_fnn_weight, w1, w2,
      w3;

  transformerWeight(size_t L)
      : rms_att_weight(L), wq(L), wk(L), wv(L), wo(L), rms_fnn_weight(L), w1(L),
        w2(L), w3(L) {}
};

transformerWeight weightLoader(std::ifstream &file, Config &headerConfig) {
  int head_size{headerConfig.dim / headerConfig.n_heads};
  int L = headerConfig.n_layers;
  transformerWeight weight(L);

  weight.emb = readTensor(file, {headerConfig.vocab_size, headerConfig.dim});
  for (int i{0}; i < L; ++i) {
    weight.rms_att_weight[i] = readTensor(file, {headerConfig.dim});
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
    weight.rms_fnn_weight[i] = readTensor(file, {headerConfig.dim});
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

  weight.rms_final_weight = readTensor(file, {headerConfig.dim});
  Tensor freq_real = readTensor(file, {headerConfig.seq_len, head_size / 2});
  Tensor freq_imag = readTensor(file, {headerConfig.seq_len, head_size / 2});

  return weight;
};

std::vector<float> embLookup(unsigned int token_id, const Tensor &embedding,
                             int dim = 288) {
  std::vector<float> result(embedding.data.begin() + (token_id * dim),
                            embedding.data.begin() + (token_id * dim) + 288);
  return result;
};

int main() {
  Tensor emb, rms_final_weight;

  std::ifstream file("stories15M.bin", std::ios_base::binary);
  Config headerConfig = readHeader(file);
  std::cout << "Here is the dimension of the file: " << headerConfig.dim
            << '\n';

  transformerWeight weight = weightLoader(file, headerConfig);

  std::cout << "stream position: " << file.tellg() << '\n';

  for (int i{0}; i < 10; ++i) {
    std::cout << weight.emb.data[i] << '\n';
  };

  std::cout << " Checking Result " << '\n';
  std::vector<float> emb_result{embLookup(0, weight.emb)};
  for (int i{0}; i < 10; ++i) {
    std::cout << emb_result[i] << '\n';
  }
}
