#pragma once
#include "tensor.h"

Tensor matmul(const Tensor &A, const Tensor &B);
Tensor matadd(const Tensor &A, const Tensor &B);
Tensor SiLU(const Tensor &A);
void softmax(const Tensor &A);
Tensor transpose(const Tensor &A);
void rmsnorm(const Tensor &A, const Tensor&W);



