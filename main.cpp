#include "ops.h"
#include <fstream>
#include <ios>
#include <iostream>
#include <string>

void readHeader(std::string fileName) {
  std::ifstream fileModel(fileName, std::ios_base::binary);
  if (!fileModel) {
    std::cout << " Error opening the file " << '\n';
    return;
  }
  int count{0};
  int header;

  while (count < 7) {
    fileModel.read(reinterpret_cast<char *>(&header), sizeof(header));
    std::cout << header << '\n';
    count++;
  }
}

int main() {
  Tensor tensor_test2{{1, 2, 3, 4}, {2, 2}};
  Tensor weight{{1, 1, 1, 1}, {1, 4}};
  rmsnorm(tensor_test2, weight);
  readHeader("stories15M.bin");
}
