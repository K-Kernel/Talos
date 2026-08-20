#include "ops.hpp"
#include "tensor.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>

std::string decode(const std::string &piece) {
  if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>')
    return std::string(
        1, static_cast<char>(std::stoi(piece.substr(3, 2), nullptr, 16)));
  return piece;
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

  const int N{256};
  std::string out;
  int token{1};

  auto t0{std::chrono::steady_clock::now()};
  for (int pos{0}; pos < N; ++pos) {
    Tensor logits{
        foward(token, pos, weight, headerConfig, key_cache, value_cache)};
    token = std::max_element(logits.data.begin(), logits.data.end()) -
            logits.data.begin();
    if (token == 1) {
      break;
    }
    out += vocab[token];
  }
  auto t1{std::chrono::steady_clock::now()};
  double sec{std::chrono::duration<double>(t1 - t0).count()};
  std::cout << out << "\n\n" << N / sec << "tok/s\n";

  printProfile(N);
}
