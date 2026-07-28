#include "ops.h"
#include "tensor.h"
#include <iostream>

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
  std::vector<float> emb_result{embLookup(0, weight.emb, headerConfig.dim)};
  for (int i{0}; i < 10; ++i) {
    std::cout << emb_result[i] << '\n';
  }

  std::cout << "Check the rotation vector function" << '\n';
  std::vector<float> test{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto result = RoPE(test, 10);
  for (float x : result) {
    std::cout << x << "";
  }
}
