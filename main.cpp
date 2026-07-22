#include "ops.h"
#include <iostream>

int main() {
  Tensor tensor_test{{101, 102, 103, 104, 105, 106}, {2, 3}};
  transpose(tensor_test);
  Tensor tensor_test2{{1, 2, 3, 4}, {2, 2}};
  Tensor weight{{1, 1, 1, 1}, {1, 4}};
  rmsnorm(tensor_test2, weight);
}
