#pragma once
#include "tensor.h"
#include <fstream>

Tensor matmul(const Tensor &A, const Tensor &B);
Tensor matadd(const Tensor &A, const Tensor &B);
Tensor SiLU(const Tensor &A);
void softmax(const Tensor &A);
Tensor transpose(const Tensor &A);
void rmsnorm(Tensor &T, const Tensor&W);
Config readHeader(std::ifstream &file);



