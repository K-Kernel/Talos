#include "ops.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

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
