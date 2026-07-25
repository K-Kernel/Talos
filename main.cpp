#include "ops.h"
#include <fstream>
#include <ios>
#include <iostream>

int main() {
  Tensor tensor_test2{{1, 2, 3, 4}, {2, 2}};
  Tensor weight{{1, 1, 1, 1}, {1, 4}};
  rmsnorm(tensor_test2, weight);

  std::ifstream file("stories15M.bin", std::ios_base::binary);
  auto headerConfig = readHeader(file);
  std::cout << "Here is the dimension of the file: " << headerConfig.dim
            << '\n';

  Tensor emb;
  emb.shape = {headerConfig.vocab_size, headerConfig.dim};
  emb.data.resize(static_cast<size_t>(headerConfig.vocab_size) *
                  headerConfig.dim);
  file.read(reinterpret_cast<char *>(emb.data.data()),
            emb.data.size() * sizeof(float));

  std::cout << "stream position: " << file.tellg() << '\n';

  for (int i{0}; i < 10; ++i) {
    std::cout << emb.data[i] << '\n';
  };
}
