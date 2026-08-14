#pragma once
#include "tensor.hpp"
#include <fstream>

Tensor matmul(const Tensor &A, const Tensor &B);
Tensor matadd_elementwise(const Tensor &A, const Tensor &B);
Tensor SiLU(const Tensor &A);
void softmax(Tensor &A);
void rmsnorm(Tensor &T, const Tensor&W);
Config readHeader(std::ifstream &file);
Tensor readTensor(std::ifstream& file , std::vector<int> Tshape);
transformerWeight weightLoader(std::ifstream &file, Config &headerConfig);
std::vector<float> embLookup(unsigned int token_id, const Tensor &embedding, int dim);
void RoPE(std::vector<float> &buffer, int head_start, int head_size,int position);
Tensor matmul_elementwise(const Tensor&A , const Tensor&B);
Tensor foward(int token, int pos, const transformerWeight &weight, const Config &config, std::vector<Tensor> &key_cache,std::vector<Tensor> &value_cache);
