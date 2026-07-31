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
Tensor readTensor(std::ifstream& file , std::vector<int> Tshape);
transformerWeight weightLoader(std::ifstream &file, Config &headerConfig);
std::vector<float> embLookup(unsigned int token_id, const Tensor &embedding, int dim);
void RoPE(std::vector<float> &buffer, int head_start, int head_size,int position);
