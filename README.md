# The tales of Talos
An LLM inference engine written from scratch in C++. No PyTorch , no BLAS every operation implemented by hand.

Run Llama-2 architecture models. Currently: Stories15M (15M params, TinyStories)

## Example
$ ./ElderScroll

Once upon a time, there was a little girl named Lily...
## Status
- [x] Tensor(flat, row-major)
- [x] Matrix multiplication, matrix sum and Sigmoid Linear unit 
- [x] Transpose , RSMNorm 
- [x] Weight loading 
- [x] Tokenizer
- [x] forward pass -> text generation
- [x] KV cache
- [ ] Benchmark
- [ ] optimise 
- [ ] quantization, CUDA

## Build

```
    git clone ...
    cmake -B build
    cmake --build build
    #download stories15M.bin and tokenizer.bin into the project root:
    # https://github.com/karpathy/llama2.c
    ./build ElderScrool
```

## How it works
A token id becomes a 288-dim vector via embedding lookup. That vector passes through 6 identical blocks: RMSNorm, then multi-head attention (6 heads of 48 dims, with rotary position embeddings applied to queries and keys), then RMSNorm again, then a SwiGLU feed-forward. Each sub-block adds its output back into the residual stream. After the final block, a last RMSNorm and a matmul against the (shared) embedding table produce 32,000 logits, and argmax picks the next token.
All tensors are flat row-major std::vector<float> with an explicit shape — one contiguous buffer, so weight loading is a single read() per tensor. Keys and values are cached per layer across positions, so each generation step only computes the current token.

## Performance
Baseline before any optimisation:

|confing| Token/sec| ms/token |GFLOP/s|
|---|---|---|---|
|stories15M, single-threaded, -03, Apple M-series | 34.3| 29.2 | 1.40|
|natural-layout matvec| 154.6 | 6.47 | 4.70|

measured over 256 greedy-decoded tokens, median of 4 runs (spread < 1%).
Weight loading excluded. 30MFLOPs/token ->  ~1.04 GFLOP/s
Transpose takes : 0.008915 seconds

Replaced per-token weight transposes with a matvec reading weights in their natural (out, in) layout — removes ~61 MB/token of copying and makes the inner loop vectorizable

## Devlogs
- [The bug that wrote over a string's guts](docs/devlog/01-tokenizer-bug.md)
- [Wrong prediction](docs/devlog/02-wrong-prediction.md)
