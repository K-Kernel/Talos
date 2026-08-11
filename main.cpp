#include "ops.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

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
    Tensor q = matmul(T_prime, transpose(weight.wq[layer]));
    Tensor k = matmul(T_prime, transpose(weight.wk[layer]));
    Tensor v = matmul(T_prime, transpose(weight.wv[layer]));

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
    Tensor att_out = matmul(output, transpose(weight.wo[layer]));
    T = matadd_elementwise(T, att_out);

    Tensor T_ffn{T};
    rmsnorm(T_ffn, weight.rms_fnn_weight[layer]);

    Tensor gate = matmul(T_ffn, transpose(weight.w1[layer]));
    Tensor up = matmul(T_ffn, transpose(weight.w3[layer]));

    gate = SiLU(gate);
    Tensor gated{matmul_elementwise(gate, up)};

    Tensor ffn_out = matmul(gated, transpose(weight.w2[layer]));
    T = matadd_elementwise(T, ffn_out);
  }

  rmsnorm(T, weight.rms_final_weight);
  return matmul(T, transpose(weight.emb)); // logits tensor
}

int main() {
  Tensor emb, rms_final_weight;

  std::ifstream file("stories15M.bin", std::ios_base::binary);
  std::ifstream tokenizer("tokenizer.bin", std::ios_base::binary);

  Config headerConfig = readHeader(file);

  transformerWeight weight = weightLoader(file, headerConfig);

  std::vector<Tensor> key_cache(headerConfig.n_layers),
      value_cache(headerConfig.n_layers);

  int max_token_length;
  tokenizer.read(reinterpret_cast<char *>(&max_token_length),
                 sizeof(max_token_length));
  std::vector<std::string> vocab(32000);

  std::cout << max_token_length << '\n';
  for (size_t i{0}; i < vocab.size(); ++i) {
    float score;
    int len;

    tokenizer.read(reinterpret_cast<char *>(&score), sizeof(score));
    tokenizer.read(reinterpret_cast<char *>(&len), sizeof(len));

    std::string s(len, '\0');
    tokenizer.read(reinterpret_cast<char *>(&s[0]), len);
    vocab[i] = s;
  }

  for (int l = 0; l < headerConfig.n_layers; ++l) {
    int n{headerConfig.seq_len * headerConfig.dim};
    key_cache[l] = Tensor{std::vector<float>(n, 0),
                          {headerConfig.seq_len, headerConfig.dim}};
    value_cache[l] = Tensor{std::vector<float>(n, 0),
                            {headerConfig.seq_len, headerConfig.dim}};
  }

  int token{1};
  for (int pos{0}; pos < 100; ++pos) {
    Tensor logits{
        foward(token, pos, weight, headerConfig, key_cache, value_cache)};
    int next = std::max_element(logits.data.begin(), logits.data.end()) -
               logits.data.begin();
    std::cout << vocab[next] << std::flush;
    token = next;
  }
}
