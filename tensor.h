#pragma once
#include <vector>

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  int row() const { return shape[0]; };
  int column() const { return shape[1]; };

  // one is for read purposes only and the other is for write
  float at(int r, int c) const { return data[r * column() + c]; }
  float &at(int r, int c) { return data[r * column() + c]; }
};

struct Config {int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;};
