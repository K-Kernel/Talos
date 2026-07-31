#include "ops.h"
#include "tensor.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  Tensor emb, rms_final_weight;

  std::ifstream file("stories15M.bin", std::ios_base::binary);

  Config headerConfig = readHeader(file);

  transformerWeight weight = weightLoader(file, headerConfig);
  std::cout << "Here is the value of the column: "
            << weight.rms_att_weight[0].row() << '\n';

  Tensor x;
  x.data = embLookup(48, weight.emb, headerConfig.dim);
  x.shape = {1, headerConfig.dim};

  int layer = 0;

  Tensor x_copy{x};
  rmsnorm(x_copy, weight.rms_att_weight[layer]);

  Tensor q = matmul(x_copy, transpose(weight.wq[layer]));
  Tensor k = matmul(x_copy, transpose(weight.wk[layer]));
  Tensor v = matmul(x_copy, transpose(weight.wv[layer]));

  assert(q.column() == 288);
  assert(q.data.size() == 288);

  // RoPE the split heads
  int head_size{headerConfig.dim / headerConfig.n_heads};
  for (int h{0}; h < headerConfig.n_heads; ++h) {
    RoPE(q.data, head_size * h, head_size, 0);
    RoPE(k.data, head_size * h, head_size, 0);
    RoPE(v.data, head_size * h, head_size, 0);
  };
}
