#include "ops.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

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

  for (int layer{0}; layer < headerConfig.n_layers; ++layer) {
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
    };

    int pos{0};
    Tensor output{std::vector<float>(headerConfig.dim, 0),
                  {1, headerConfig.dim}};

    for (int h{0}; h < headerConfig.n_heads; ++h) {
      Tensor scores{std::vector<float>(pos + 1, 0), {1, pos + 1}};

      for (int p{0}; p <= pos; ++p) {
        float dot{0};
        for (int i{0}; i < head_size; ++i) {
          dot += q.data[h * head_size + i] * k.data[h * head_size + i];
        }
        scores.data[p] = dot / std::sqrt(head_size);
      }

      softmax(scores);
      std::cout << "The sum is "
                << std::accumulate(scores.data.begin(), scores.data.end(), 0.0f)
                << '\n';

      for (int d{0}; d < head_size; ++d) {
        float sum = 0;
        for (int p{0}; p <= pos; ++p) {
          sum += scores.data[p] * v.data[h * head_size + d];
        }
        output.data[h * head_size + d] = sum;
      }
    }

    Tensor att_out = matmul(output, transpose(weight.wo[layer]));
    x = matadd_elementwise(x, att_out);

    // Attetion finished
    Tensor xb = x;
    rmsnorm(xb, weight.rms_fnn_weight[layer]);

    Tensor gate = matmul(xb, transpose(weight.w1[layer]));
    Tensor up = matmul(xb, transpose(weight.w3[layer]));

    gate = SiLU(gate);
    Tensor gated = matmul_elementwise(gate, up);

    Tensor ffn_out = matmul(gated, transpose(weight.w2[layer]));
    x = matadd_elementwise(x, ffn_out);

    rmsnorm(x, weight.rms_final_weight);
    Tensor logits = matmul(x, transpose(weight.emb));
    int next_token = std::max_element(logits.data.begin(), logits.data.end()) -
                     logits.data.begin();
    std::cout << "next token id: " << next_token << '\n';
  }
}
