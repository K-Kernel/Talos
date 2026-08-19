# I predicted 1.2x. I got 4.5x.

The engine was finally working, and with that, a new mission emerged: optimisation.

The first step for optimisation is benchmarking; you need a baseline to compare to. I rebuilt my Cmake file, creating two modes: release mode and debug mode. In release, I use the flag ```-O3```, that means optimisation level 3; it tells the compiler that it is allowed to aggressively optimise the programme. For debug, I have AddressSanitiser (Asan); it is very helpful to pinpoint where crashes or overflows happen,  saving debugging time. This split needs to be made because Asan can significantly slow down the programme  because it adds extra memory checks. 

Before any optimisation work, I need a baseline to compare against; I measure the time using ```std::chrono::steady_clock::now()```, steady_clock is a good choice because it is designed for measuring elapsed time. It isn't affected by the system clock being adjusted. The baseline result was 34.3 toks/s, the median of 3 runs with one run for warm-up. The run spread was less than 1%. 

My inference engine does around 15M multiply accumulate per token. At 15M MAC this is equivalent to 30MFLOPs per token, that result in the  34.3 toks/s meaning the program was doing 1.04 GFLOP/s. Naive scalar code on one core do 1-3 GFLOP/s, so I was sitting at the floor, transposing was moving exactly 15M the same amount as matrix multiplication except it was not doing any arithmetic. Now when it comes to optimisation, experience and a well-structured reasoning can give you the intuition or an educated guess of where the problem might be, but there is no way to know for sure; that's when we need to measure it. I measured transpose of the embedding and it took 30% of the total time. 


There were two fixes: transpose every weight at load time, dropping all ```transpose()``` from the ```forward()``` function, or the cleaner solution was to write a function that takes the embedding and the weights in its natural ```out,in``` layout. The key idea is that transpose then multiply is the same as multiply while reading the matrix in a different index order.

## Matrix multiplication before
```
Tensor matmul(const Tensor &A, const Tensor &B) {
  Tensor result{std::vector<float>(A.row() * B.column(), 0),
                {A.shape[0], B.shape[1]}};
  for (int i{0}; i < A.row(); ++i) {
    for (int j{0}; j < B.column(); ++j) {
      for (int k{0}; k < A.column(); ++k) {
        result.at(i, j) += A.at(i, k) * B.at(k, j);
      }
    }
  }

  return result;
}
```

## Matrix multiplication after

```
Tensor matvec(const Tensor &A, const Tensor &W) {
  assert(static_cast<int>(A.data.size()) == W.column());
  Tensor result{std::vector<float>(W.row(), 0), {1, W.row()}};

  for (int i{0}; i < W.row(); i++) {
    float sum{0};
    for (int k{0}; k < W.column(); ++k) {
      sum += W.at(i, k) * A.data[k];
    }
    result.data[i] = sum;
  }

  return result;
}

```

After applying this change instead of doing the matrix multiplication of X times transpose T, I just computed the dot product of row j of W with x that it’s the same. The new result was 154.6 tok/s, I was predicting 1.2x, and got 4.5x.

| | tok/s | ms/token | GFLOP/s |
|---|---|---|---|
|baseline| 34.3|29.2| 1.04| 
|natural layout matvec| 154.6|6.47| 4.70| 


The reason for this big improvement was the way matrix multiplication was structured , ``` B.at(k,j)``` could jump 288 or 768 floats, the compiler can't vectorise a strided load like that, whereas ```w.at(i,k)``` was accessing consecutive chunks of memory allowing the compiler to emit NEON and read four floats per instruction. Transpose didn't just slow down my program, it was also blocking optimisation.

But this teaches me a valuable lesson: memory layout isn't what you do afterwards, it decides which optimisations the compiler is permitted to attempt at all. 

Then I deleted a dead function and got 16% faster. The old commit gave 154.8 tok/s, the new one 179.3, run minutes apart on the same machine. Nothing that executed had changed — removing matmul shifted every later function's address in the binary, and a hot loop's alignment in the instruction cache is worth that much. Which means my noise floor is around 16%, and any future "optimisation" smaller than that isn't measurable without controlling for layout.
