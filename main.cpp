#include "ops.h"
#include "tensor.h"
#include <iostream>
#include <vector>

int main() {
  Tensor emb, rms_final_weight;

  std::ifstream file("stories15M.bin", std::ios_base::binary);
  auto headerConfig = readHeader(file);
  std::cout << "Here is the dimension of the file: " << headerConfig.dim
            << '\n';

  emb = readTensor(file, {headerConfig.vocab_size, headerConfig.dim});

  int head_size{headerConfig.dim / headerConfig.n_heads};
  int L = headerConfig.n_layers;
  std::vector<Tensor> rms_att_weight(L), wq(L), wk(L), wv(L), wo(L),
      rms_fnn_weight(L), w1(L), w2(L), w3(L);

  for (int i{0}; i < L; ++i) {
    rms_att_weight[i] = readTensor(file, {headerConfig.dim});
  };
  for (int i{0}; i < L; ++i) {
    wq[i] = readTensor(file, {headerConfig.dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    wk[i] = readTensor(file,
                       {headerConfig.dim, headerConfig.n_kv_heads * head_size});
  }
  for (int i{0}; i < L; ++i) {
    wv[i] = readTensor(file,
                       {headerConfig.dim, headerConfig.n_kv_heads * head_size});
  }
  for (int i{0}; i < L; ++i) {
    wo[i] = readTensor(file, {headerConfig.dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    rms_fnn_weight[i] = readTensor(file, {headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    w1[i] = readTensor(file, {headerConfig.hidden_dim, headerConfig.dim});
  }
  for (int i{0}; i < L; ++i) {
    w2[i] = readTensor(file, {headerConfig.dim, headerConfig.hidden_dim});
  }
  for (int i{0}; i < L; ++i) {
    w3[i] = readTensor(file, {headerConfig.hidden_dim, headerConfig.dim});
  }

  rms_final_weight = readTensor(file, {headerConfig.dim});
  Tensor freq_real = readTensor(file, {headerConfig.seq_len, head_size / 2});
  Tensor freq_imag = readTensor(file, {headerConfig.seq_len, head_size / 2});
  std::cout << "stream position: " << file.tellg() << '\n';

  for (int i{0}; i < 10; ++i) {
    std::cout << rms_att_weight[1].data[i] << '\n';
  };
}
